#include "qemu/osdep.h"
#include "qemu/log.h"
#include "qemu/module.h"
#include "qapi/error.h"
#include "qom/object.h"
#include "hw/registerfields.h"
#include "cpu.h"
#include "exec/address-spaces.h"
#include "exec/cpu-common.h"
#include "target/arm/cpu.h"
#include "exec/exec-all.h"
#include "qapi/visitor.h"
#include "hw/qdev-properties.h"
#include "hw/irq.h"
#include "hw/arm/mmio_proxy.h"
#include "exec/memory.h"
#include "exec/memop.h"
#include "hw/sysbus.h"
#include "inttypes.h"
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <errno.h>
#include <sys/mman.h>
#include <fcntl.h>

MMIOProxyState *global_mmio_proxy = NULL;

// 获取当前时间戳（微秒）
static uint64_t get_timestamp_us(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000ULL + tv.tv_usec;
}

// 初始化共享内存
static bool init_shared_memory(MMIOProxyState *s)
{
    if (!s->shared_mem_name) {
        s->shared_mem_name = g_strdup("/mmio_proxy_shared");
    }
    
    // 创建或打开共享内存
    s->shared_mem_fd = shm_open(s->shared_mem_name, O_CREAT | O_RDWR, 0666);
    if (s->shared_mem_fd == -1) {
        qemu_log("Failed to create shared memory: %s\n", strerror(errno));
        return false;
    }
    
    // 设置共享内存大小
    if (ftruncate(s->shared_mem_fd, SHARED_MEM_SIZE) == -1) {
        qemu_log("Failed to set shared memory size: %s\n", strerror(errno));
        close(s->shared_mem_fd);
        return false;
    }
    
    // 映射共享内存
    s->shared_log = mmap(NULL, SHARED_MEM_SIZE, PROT_READ | PROT_WRITE, 
                        MAP_SHARED, s->shared_mem_fd, 0);
    if (s->shared_log == MAP_FAILED) {
        qemu_log("Failed to map shared memory: %s\n", strerror(errno));
        close(s->shared_mem_fd);
        return false;
    }
    
    // 初始化共享内存结构
    memset(s->shared_log, 0, SHARED_MEM_SIZE);
    s->shared_log->entry_count = 0;
    s->shared_log->write_index = 0;
    
    qemu_log("Shared memory initialized: %s\n", s->shared_mem_name);
    return true;
}

// 记录状态到共享内存
void mmio_proxy_log_state(MMIOProxyState *s, uint64_t addr, uint64_t val, 
                         uint32_t size, bool is_write, int irq_num)
{
    if (!s || !s->shared_log) {
        return;
    }
    
    // 获取当前CPU状态
    CPUState *cs = first_cpu;
    if (!cs || !object_dynamic_cast(OBJECT(cs), TYPE_ARM_CPU)) {
        return;
    }
    
    ARMCPU *cpu = ARM_CPU(cs);
    CPUARMState *env = &cpu->env;
    
    // 获取写入位置
    uint32_t index = s->shared_log->write_index;
    StateLogEntry *entry = &s->shared_log->entries[index];
    
    // 填充状态信息
    entry->timestamp = get_timestamp_us();
    entry->cpu_id = cs->cpu_index;
    entry->irq_num = irq_num;
    entry->pc = env->pc;
    entry->sp = env->sp_el[0];
    
    // 复制寄存器状态
    for (int i = 0; i < 31; i++) {
        entry->xregs[i] = env->xregs[i];
    }
    
    // MMIO访问信息
    entry->mmio_addr = addr;
    entry->mmio_val = val;
    entry->mmio_size = size;
    entry->is_write = is_write ? 1 : 0;
    
    // 复制外设寄存器状态
    if (s->regs && s->size <= 256) {
        memcpy(entry->mmio_regs, s->regs, s->size);
    }
    
    // 更新索引
    s->shared_log->write_index = (index + 1) % MAX_LOG_ENTRIES;
    if (s->shared_log->entry_count < MAX_LOG_ENTRIES) {
        s->shared_log->entry_count++;
    }
    
    qemu_log("State logged: addr=0x%lx, val=0x%lx, size=%u, %s, PC=0x%lx\n",
             addr, val, size, is_write ? "WRITE" : "READ", entry->pc);
}

// 代理读操作
static uint64_t mmio_proxy_read(void *opaque, hwaddr addr, unsigned size)
{
    MMIOProxyState *s = opaque;
    uint64_t val = 0;
    
    qemu_log("MMIO Proxy READ: addr=0x%lx, size=%u\n", addr, size);
    
    // 如果有目标设备，从目标设备读取
    if (s->target_region) {
        MemTxResult result;
        result = memory_region_dispatch_read(s->target_region, addr, &val, 
                                           size_memop(size) | MO_TE, 
                                           MEMTXATTRS_UNSPECIFIED);
        if (result != MEMTX_OK) {
            qemu_log("Target device read failed\n");
        }
    } else if (s->regs && addr + size <= s->size) {
        // 从本地缓存读取
        memcpy(&val, &s->regs[addr], size);
    }
    
    // 更新本地缓存
    if (s->regs && addr + size <= s->size) {
        memcpy(&s->regs[addr], &val, size);
    }
    
    // 记录状态到共享内存
    mmio_proxy_log_state(s, s->base + addr, val, size, false, -1);
    
    return val;
}

// 代理写操作
static void mmio_proxy_write(void *opaque, hwaddr addr, uint64_t val, unsigned size)
{
    MMIOProxyState *s = opaque;
    
    qemu_log("MMIO Proxy WRITE: addr=0x%lx, val=0x%lx, size=%u\n", addr, val, size);
    
    // 如果有目标设备，写入目标设备
    if (s->target_region) {
        MemTxResult result;
        result = memory_region_dispatch_write(s->target_region, addr, val,
                                            size_memop(size) | MO_TE,
                                            MEMTXATTRS_UNSPECIFIED);
        if (result != MEMTX_OK) {
            qemu_log("Target device write failed\n");
        }
    }
    
    // 更新本地缓存
    if (s->regs && addr + size <= s->size) {
        memcpy(&s->regs[addr], &val, size);
    }
    
    // 记录状态到共享内存
    mmio_proxy_log_state(s, s->base + addr, val, size, true, -1);
}

static const MemoryRegionOps mmio_proxy_ops = {
    .read = mmio_proxy_read,
    .write = mmio_proxy_write,
    .endianness = DEVICE_NATIVE_ENDIAN,
    .valid = {
        .min_access_size = 1,
        .max_access_size = 8,
    },
};

// 查找并设置目标设备
static bool setup_target_device(MMIOProxyState *s)
{
    if (!s->target_id) {
        qemu_log("No target device specified\n");
        return false;
    }
    
    // 通过路径查找设备
    Object *target = object_resolve_path(s->target_id, NULL);
    if (!target) {
        qemu_log("Target device '%s' not found\n", s->target_id);
        return false;
    }
    
    if (!object_dynamic_cast(target, TYPE_DEVICE)) {
        qemu_log("Target '%s' is not a device\n", s->target_id);
        return false;
    }
    
    s->target_device = DEVICE(target);
    
    // 如果是SysBus设备，获取其MMIO region
    if (object_dynamic_cast(target, TYPE_SYS_BUS_DEVICE)) {
        SysBusDevice *sbd = SYS_BUS_DEVICE(target);
        s->target_region = sysbus_mmio_get_region(sbd, 0);
        if (s->target_region) {
            qemu_log("Found target MMIO region for %s\n", s->target_id);
            return true;
        }
    }
    
    qemu_log("Could not get MMIO region for target device\n");
    return false;
}

static void mmio_proxy_realize(DeviceState *dev, Error **errp)
{
    MMIOProxyState *s = MMIO_PROXY(dev);
    
    if (!s->size) {
        s->size = 0x1000;  // 默认4KB
    }
    
    qemu_log("MMIO Proxy realize: base=0x%lx, size=0x%lx\n", 
             (unsigned long)s->base, (unsigned long)s->size);
    
    // 分配寄存器缓存
    s->regs = g_malloc0(s->size);
    if (!s->regs) {
        error_setg(errp, "Failed to allocate register cache");
        return;
    }
    
    // 初始化共享内存
    if (!init_shared_memory(s)) {
        error_setg(errp, "Failed to initialize shared memory");
        return;
    }
    
    // 设置目标设备
    setup_target_device(s);
    
    // 创建MMIO区域
    memory_region_init_io(&s->iomem, OBJECT(dev), &mmio_proxy_ops, s,
                          TYPE_MMIO_PROXY, s->size);
    
    // 将MMIO区域添加到系统内存
    memory_region_add_subregion_overlap(get_system_memory(), s->base, 
                                       &s->iomem, 10); // 高优先级
    
    // 解析IRQ列表
    if (s->irq_list_str) {
        char *irq_list_dup = g_strdup(s->irq_list_str);
        char *token = strtok(irq_list_dup, ";,");
        s->num_monitored_irqs = 0;
        while (token && s->num_monitored_irqs < MAX_MONITORED_IRQS) {
            s->monitored_irqs[s->num_monitored_irqs++] = atoi(token);
            qemu_log("Monitoring IRQ: %d\n", s->monitored_irqs[s->num_monitored_irqs-1]);
            token = strtok(NULL, ";,");
        }
        g_free(irq_list_dup);
    }
    
    // 设置全局指针
    global_mmio_proxy = s;
    
    qemu_log("MMIO Proxy device realized successfully\n");
}

static void mmio_proxy_finalize(Object *obj)
{
    MMIOProxyState *s = MMIO_PROXY(obj);
    
    // 清理共享内存
    if (s->shared_log && s->shared_log != MAP_FAILED) {
        munmap(s->shared_log, SHARED_MEM_SIZE);
    }
    if (s->shared_mem_fd >= 0) {
        close(s->shared_mem_fd);
    }
    if (s->shared_mem_name) {
        shm_unlink(s->shared_mem_name);
        g_free(s->shared_mem_name);
    }
    
    // 清理其他资源
    g_free(s->regs);
    g_free(s->irq_list_str);
    g_free(s->target_id);
}

static Property mmio_proxy_properties[] = {
    DEFINE_PROP_UINT64("base", MMIOProxyState, base, 0),
    DEFINE_PROP_UINT64("size", MMIOProxyState, size, 0x1000),
    DEFINE_PROP_STRING("irq", MMIOProxyState, irq_list_str),
    DEFINE_PROP_STRING("target", MMIOProxyState, target_id),
    DEFINE_PROP_STRING("shared_mem", MMIOProxyState, shared_mem_name),
    DEFINE_PROP_END_OF_LIST(),
};

static void mmio_proxy_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    dc->realize = mmio_proxy_realize;
    device_class_set_props(dc, mmio_proxy_properties);
    dc->desc = "MMIO Proxy Device for peripheral monitoring and emulation";
}

static const TypeInfo mmio_proxy_info = {
    .name          = TYPE_MMIO_PROXY,
    .parent        = TYPE_DEVICE,
    .instance_size = sizeof(MMIOProxyState),
    .class_init    = mmio_proxy_class_init,
    .instance_finalize = mmio_proxy_finalize,
};

static void mmio_proxy_register_types(void)
{
    type_register_static(&mmio_proxy_info);
}

// IRQ hook函数
void mmio_proxy_dump_on_irq(void) {
    if (!global_mmio_proxy) {
        return;
    }
    
    MMIOProxyState *s = global_mmio_proxy;
    qemu_log("IRQ triggered, logging state to shared memory\n");
    
    // 记录IRQ触发时的状态
    mmio_proxy_log_state(s, 0, 0, 0, false, -1);
}

type_init(mmio_proxy_register_types);
