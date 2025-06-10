#ifndef MMIO_INFERENCE_BRIDGE_H
#define MMIO_INFERENCE_BRIDGE_H

#include "qemu/osdep.h"
#include "hw/qdev-core.h"
#include "hw/sysbus.h"
#include "exec/memory.h"
#include "qom/object.h"
#include <stdbool.h>
#include <stdint.h>

#define MAX_PERIPHERAL_DEVICES 64
#define MAX_DEVICE_NAME_LEN 128
#define SHARED_MEM_NAME_PREFIX "/mmio_inference_"

#define MAX_INFERRED_REGISTERS 64
#define INFERENCE_SHARED_MEM_SIZE 8192

// AI推断结果结构
typedef struct {
    uint64_t device_addr;           // 设备基地址
    uint32_t register_count;        // 推断出的寄存器数量
    struct {
        uint32_t offset;            // 寄存器偏移
        uint64_t value;             // 推断的值
        uint32_t confidence;        // 置信度 (0-100)
        uint32_t size;              // 寄存器大小 (1,2,4,8字节)
        char name[32];              // 寄存器名称
        char description[128];      // 描述
    } registers[MAX_INFERRED_REGISTERS];
    uint64_t timestamp;             // 推断时间戳
    uint32_t inference_id;          // 推断ID
    uint32_t need_resume;           // 是否需要恢复执行
} InferenceResult;

// 推断桥接共享内存结构
typedef struct {
    uint32_t pending_inference;     // 是否有待处理的推断请求
    uint32_t inference_complete;    // 推断是否完成
    uint64_t fault_addr;            // 触发错误的地址
    uint64_t fault_pc;              // 触发错误的PC
    uint32_t fault_size;            // 访问大小
    uint32_t is_write;              // 是否为写操作
    InferenceResult result;         // 推断结果
    uint32_t qemu_paused;           // QEMU是否已暂停
    uint32_t resume_requested;      // 是否请求恢复
} InferenceBridge;

// 外设推断设备状态
typedef struct InferenceDeviceState {
    DeviceState parent_obj;
    
    // 共享内存相关
    char *bridge_mem_name;          // 桥接共享内存名称
    int bridge_mem_fd;              // 共享内存文件描述符
    InferenceBridge *bridge;        // 桥接内存映射
    
    // 内存区域管理
    MemoryRegion *system_memory;    // 系统内存引用
    MemoryRegion *inferred_regions; // 推断出的内存区域数组
    uint32_t region_count;          // 已创建的区域数量
    
    // 错误处理相关
    bool inference_mode;            // 是否处于推断模式
    uint64_t current_fault_addr;    // 当前错误地址
    
} InferenceDeviceState;

#define TYPE_INFERENCE_DEVICE "inference-device"
OBJECT_DECLARE_SIMPLE_TYPE(InferenceDeviceState, INFERENCE_DEVICE)

// 函数声明
void inference_handle_mmio_fault(uint64_t addr, uint64_t pc, uint32_t size, bool is_write);
bool inference_apply_result(const InferenceResult *result);
void inference_resume_execution(void);
int inference_wait_for_result(uint32_t timeout_ms);

#endif // MMIO_INFERENCE_BRIDGE_H 
