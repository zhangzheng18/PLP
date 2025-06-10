#include "qemu/osdep.h"
#include "qemu/log.h"
#include "qemu/module.h"
#include "qapi/error.h"
#include "qom/object.h"
#include "hw/qdev-properties.h"
#include "hw/arm/mmio_inference_bridge.h"
#include "exec/memory.h"
#include "exec/address-spaces.h"
#include "sysemu/runstate.h"
#include "cpu.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

static InferenceDeviceState *global_inference_device = NULL;

// 初始化桥接共享内存
static bool init_bridge_memory(InferenceDeviceState *s)
{
    if (!s->bridge_mem_name) {
        s->bridge_mem_name = g_strdup("/mmio_inference_bridge");
    }
    
    // 创建或打开共享内存
    s->bridge_mem_fd = shm_open(s->bridge_mem_name, O_CREAT | O_RDWR, 0666);
    if (s->bridge_mem_fd == -1) {
        qemu_log("Failed to create inference bridge memory: %s\n", strerror(errno));
        return false;
    }
    
    // 设置共享内存大小
    if (ftruncate(s->bridge_mem_fd, INFERENCE_SHARED_MEM_SIZE) == -1) {
        qemu_log("Failed to set inference bridge memory size: %s\n", strerror(errno));
        close(s->bridge_mem_fd);
        return false;
    }
    
    // 映射共享内存
    s->bridge = mmap(NULL, INFERENCE_SHARED_MEM_SIZE, PROT_READ | PROT_WRITE, 
                    MAP_SHARED, s->bridge_mem_fd, 0);
    if (s->bridge == MAP_FAILED) {
        qemu_log("Failed to map inference bridge memory: %s\n", strerror(errno));
        close(s->bridge_mem_fd);
        return false;
    }
    
    // 初始化共享内存结构
    memset(s->bridge, 0, INFERENCE_SHARED_MEM_SIZE);
    s->bridge->pending_inference = 0;
    s->bridge->inference_complete = 0;
    s->bridge->qemu_paused = 0;
    s->bridge->resume_requested = 0;
    
    qemu_log("Inference bridge memory initialized: %s\n", s->bridge_mem_name);
    return true;
}

// 创建推断出的外设内存区域
static bool create_inferred_device(InferenceDeviceState *s, const InferenceResult *result)
{
    if (s->region_count >= MAX_INFERRED_REGISTERS) {
        qemu_log("Too many inferred regions\n");
        return false;
    }
    
    // 为新设备分配内存区域
    MemoryRegion *region = g_malloc0(sizeof(MemoryRegion));
    if (!region) {
        qemu_log("Failed to allocate memory region\n");
        return false;
    }
    
    // 计算设备大小（基于寄存器分布）
    uint64_t device_size = 0x1000; // 默认4KB
    for (uint32_t i = 0; i < result->register_count; i++) {
        uint64_t reg_end = result->registers[i].offset + result->registers[i].size;
        if (reg_end > device_size) {
            device_size = (reg_end + 0xFFF) & ~0xFFF; // 对齐到4KB边界
        }
    }
    
    // 创建设备状态存储
    uint8_t *device_state = g_malloc0(device_size);
    
    // 根据推断结果初始化寄存器值
    for (uint32_t i = 0; i < result->register_count; i++) {
        uint32_t offset = result->registers[i].offset;
        uint64_t value = result->registers[i].value;
        uint32_t size = result->registers[i].size;
        
        if (offset + size <= device_size) {
            memcpy(device_state + offset, &value, size);
            qemu_log("Initialized register %s at 0x%x = 0x%lx\n", 
                    result->registers[i].name, offset, value);
        }
    }
    
    // 创建MMIO操作函数
    static uint64_t inferred_device_read(void *opaque, hwaddr addr, unsigned size)
    {
        uint8_t *state = (uint8_t *)opaque;
        uint64_t value = 0;
        
        if (addr + size <= device_size) {
            memcpy(&value, state + addr, size);
        }
        
        qemu_log("Inferred device READ: addr=0x%lx, val=0x%lx, size=%u\n", 
                addr, value, size);
        return value;
    }
    
    static void inferred_device_write(void *opaque, hwaddr addr, uint64_t value, unsigned size)
    {
        uint8_t *state = (uint8_t *)opaque;
        
        qemu_log("Inferred device WRITE: addr=0x%lx, val=0x%lx, size=%u\n", 
                addr, value, size);
        
        if (addr + size <= device_size) {
            memcpy(state + addr, &value, size);
        }
    }
    
    static const MemoryRegionOps inferred_device_ops = {
        .read = inferred_device_read,
        .write = inferred_device_write,
        .endianness = DEVICE_NATIVE_ENDIAN,
        .valid = {
            .min_access_size = 1,
            .max_access_size = 8,
        },
    };
    
    // 初始化内存区域
    char region_name[64];
    snprintf(region_name, sizeof(region_name), "inferred-device-0x%lx", result->device_addr);
    
    memory_region_init_io(region, OBJECT(s), &inferred_device_ops, device_state,
                          region_name, device_size);
    
    // 将设备添加到系统内存
    memory_region_add_subregion(get_system_memory(), result->device_addr, region);
    
    // 保存区域信息
    if (!s->inferred_regions) {
        s->inferred_regions = g_malloc0(sizeof(MemoryRegion) * MAX_INFERRED_REGISTERS);
    }
    memcpy(&s->inferred_regions[s->region_count], region, sizeof(MemoryRegion));
    s->region_count++;
    
    qemu_log("Created inferred device at 0x%lx, size=0x%lx, %u registers\n",
             result->device_addr, device_size, result->register_count);
    
    return true;
}

// 处理MMIO访问错误
void inference_handle_mmio_fault(uint64_t addr, uint64_t pc, uint32_t size, bool is_write)
{
    if (!global_inference_device || !global_inference_device->bridge) {
        qemu_log("Inference device not available for fault handling\n");
        return;
    }
    
    InferenceDeviceState *s = global_inference_device;
    
    qemu_log("=== MMIO FAULT DETECTED ===\n");
    qemu_log("Address: 0x%lx, PC: 0x%lx, Size: %u, Operation: %s\n",
             addr, pc, size, is_write ? "WRITE" : "READ");
    
    // 检查是否已经在处理推断
    if (s->bridge->pending_inference) {
        qemu_log("Inference already pending, ignoring duplicate fault\n");
        return;
    }
    
    // 填充错误信息到共享内存
    s->bridge->fault_addr = addr;
    s->bridge->fault_pc = pc;
    s->bridge->fault_size = size;
    s->bridge->is_write = is_write ? 1 : 0;
    s->bridge->pending_inference = 1;
    s->bridge->inference_complete = 0;
    s->bridge->qemu_paused = 1;
    
    // 记录当前时间戳
    struct timeval tv;
    gettimeofday(&tv, NULL);
    uint64_t timestamp = tv.tv_sec * 1000000ULL + tv.tv_usec;
    
    qemu_log("Inference request created at timestamp %lu\n", timestamp);
    qemu_log("Waiting for AI inference daemon to process...\n");
    
    // 启动等待推断结果的线程（非阻塞）
    qemu_log("QEMU will continue running while waiting for inference\n");
}

// 应用推断结果
bool inference_apply_result(const InferenceResult *result)
{
    if (!global_inference_device) {
        qemu_log("No inference device available\n");
        return false;
    }
    
    InferenceDeviceState *s = global_inference_device;
    
    qemu_log("Applying inference result for device 0x%lx with %u registers\n",
             result->device_addr, result->register_count);
    
    // 创建推断出的设备
    if (!create_inferred_device(s, result)) {
        qemu_log("Failed to create inferred device\n");
        return false;
    }
    
    // 标记推断完成
    if (s->bridge) {
        s->bridge->inference_complete = 1;
        s->bridge->pending_inference = 0;
        memcpy(&s->bridge->result, result, sizeof(InferenceResult));
    }
    
    qemu_log("Inference result applied successfully\n");
    return true;
}

// 等待推断结果
int inference_wait_for_result(uint32_t timeout_ms)
{
    if (!global_inference_device || !global_inference_device->bridge) {
        return -1;
    }
    
    InferenceBridge *bridge = global_inference_device->bridge;
    uint32_t elapsed = 0;
    const uint32_t poll_interval = 100; // 100ms
    
    while (elapsed < timeout_ms) {
        if (bridge->inference_complete) {
            // 应用推断结果
            inference_apply_result(&bridge->result);
            return 0;
        }
        
        g_usleep(poll_interval * 1000);
        elapsed += poll_interval;
    }
    
    qemu_log("Inference timeout after %u ms\n", timeout_ms);
    return -1;
}

// 恢复执行
void inference_resume_execution(void)
{
    if (!global_inference_device || !global_inference_device->bridge) {
        return;
    }
    
    InferenceDeviceState *s = global_inference_device;
    
    s->bridge->qemu_paused = 0;
    s->bridge->resume_requested = 1;
    
    // 恢复QEMU执行
    qemu_system_wakeup_request(QEMU_WAKEUP_REASON_OTHER, NULL);
    
    qemu_log("QEMU execution resumed\n");
}

// 设备实现
static void inference_device_realize(DeviceState *dev, Error **errp)
{
    InferenceDeviceState *s = INFERENCE_DEVICE(dev);
    
    qemu_log("Initializing inference device\n");
    
    // 初始化桥接共享内存
    if (!init_bridge_memory(s)) {
        error_setg(errp, "Failed to initialize inference bridge memory");
        return;
    }
    
    // 设置全局指针
    global_inference_device = s;
    
    qemu_log("Inference device initialized successfully\n");
}

static void inference_device_finalize(Object *obj)
{
    InferenceDeviceState *s = INFERENCE_DEVICE(obj);
    
    // 清理共享内存
    if (s->bridge && s->bridge != MAP_FAILED) {
        munmap(s->bridge, INFERENCE_SHARED_MEM_SIZE);
    }
    if (s->bridge_mem_fd >= 0) {
        close(s->bridge_mem_fd);
    }
    if (s->bridge_mem_name) {
        shm_unlink(s->bridge_mem_name);
        g_free(s->bridge_mem_name);
    }
    
    // 清理其他资源
    g_free(s->inferred_regions);
}

static Property inference_device_properties[] = {
    DEFINE_PROP_STRING("bridge_mem", InferenceDeviceState, bridge_mem_name),
    DEFINE_PROP_END_OF_LIST(),
};

static void inference_device_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    dc->realize = inference_device_realize;
    device_class_set_props(dc, inference_device_properties);
    dc->desc = "AI Inference Bridge Device for peripheral inference and emulation";
}

static const TypeInfo inference_device_info = {
    .name          = TYPE_INFERENCE_DEVICE,
    .parent        = TYPE_DEVICE,
    .instance_size = sizeof(InferenceDeviceState),
    .class_init    = inference_device_class_init,
    .instance_finalize = inference_device_finalize,
};

static void inference_device_register_types(void)
{
    type_register_static(&inference_device_info);
}

type_init(inference_device_register_types); 
