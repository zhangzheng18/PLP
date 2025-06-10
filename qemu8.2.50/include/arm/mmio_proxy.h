#ifndef MMIO_PROXY_H
#define MMIO_PROXY_H

#include <stdint.h>
#include "hw/qdev-core.h"
#include "hw/qdev-properties.h"
#include "hw/irq.h"
#include "qom/object.h"
#include "hw/sysbus.h"
#include "exec/memory.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#define MAX_MONITORED_IRQS 32
#define SHARED_MEM_SIZE 4096
#define MAX_LOG_ENTRIES 100

// 共享内存中的状态记录结构
typedef struct {
    uint64_t timestamp;
    uint32_t cpu_id;
    uint32_t irq_num;
    uint64_t pc;
    uint64_t sp;
    uint64_t xregs[31];
    uint64_t mmio_addr;
    uint64_t mmio_val;
    uint32_t mmio_size;
    uint32_t is_write;  // 0=read, 1=write
    uint8_t mmio_regs[256]; // 外设寄存器状态快照
} StateLogEntry;

typedef struct {
    uint32_t entry_count;
    uint32_t write_index;
    StateLogEntry entries[MAX_LOG_ENTRIES];
} SharedMemoryLog;

typedef struct MMIOProxyState {
    DeviceState parent_obj;
    MemoryRegion iomem;
    MemoryRegion *target_region; // 被代理外设的MMIO region
    qemu_irq proxied_irq;        // 代理后的中断线（连接到GIC）
    qemu_irq target_irq;         // 被代理外设的原始中断线
    uint64_t base, size;
    uint8_t *regs;
    char *irq_list_str;
    int monitored_irqs[MAX_MONITORED_IRQS];
    int num_monitored_irqs;
    char *target_id; // QOM id of the target device
    
    // 共享内存相关
    char *shared_mem_name;       // 共享内存名称
    int shared_mem_fd;           // 共享内存文件描述符
    SharedMemoryLog *shared_log; // 共享内存映射
    
    // 原始外设相关
    DeviceState *target_device;  // 目标设备指针
    MemoryRegion *original_region; // 原始设备的MMIO区域
    bool intercept_enabled;      // 是否启用拦截
} MMIOProxyState;

extern MMIOProxyState *global_mmio_proxy;

#define TYPE_MMIO_PROXY "mmio-proxy"
OBJECT_DECLARE_SIMPLE_TYPE(MMIOProxyState, MMIO_PROXY)

// 函数声明
void mmio_proxy_dump_on_irq(void);
void mmio_proxy_log_state(MMIOProxyState *s, uint64_t addr, uint64_t val, 
                         uint32_t size, bool is_write, int irq_num);

#endif // MMIO_PROXY_H
