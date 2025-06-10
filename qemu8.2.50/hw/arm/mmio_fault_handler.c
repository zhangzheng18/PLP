#include "qemu/osdep.h"
#include "qemu/log.h"
#include "exec/memory.h"
#include "exec/address-spaces.h"
#include "hw/arm/mmio_inference_bridge.h"
#include "cpu.h"
#include "exec/exec-all.h"

// 全局错误处理器状态
static bool fault_handler_enabled = false;
static MemTxResult (*original_memory_dispatch_read)(MemoryRegion *mr, hwaddr addr,
                                                   uint64_t *pval, MemOp op,
                                                   MemTxAttrs attrs) = NULL;
static MemTxResult (*original_memory_dispatch_write)(MemoryRegion *mr, hwaddr addr,
                                                    uint64_t val, MemOp op,
                                                    MemTxAttrs attrs) = NULL;

// 检查地址是否已映射
static bool is_address_mapped(hwaddr addr, uint64_t size)
{
    MemoryRegion *mr = address_space_translate(&address_space_memory, 
                                              addr, &addr, &size, false,
                                              MEMTXATTRS_UNSPECIFIED);
    return mr && !memory_region_is_ram(mr);
}

// MMIO读取错误处理
static MemTxResult mmio_read_fault_handler(MemoryRegion *mr, hwaddr addr,
                                          uint64_t *pval, MemOp op,
                                          MemTxAttrs attrs)
{
    uint64_t size = memop_size(op);
    
    // 检查是否为未映射的MMIO访问
    if (!is_address_mapped(addr, size)) {
        CPUState *cpu = current_cpu;
        uint64_t pc = 0;
        
        if (cpu) {
            CPUClass *cc = CPU_GET_CLASS(cpu);
            if (cc->get_pc) {
                pc = cc->get_pc(cpu);
            }
        }
        
        qemu_log("MMIO Read Fault: addr=0x%lx, size=%lu, PC=0x%lx\n", 
                addr, size, pc);
        
        // 触发推断处理
        inference_handle_mmio_fault(addr, pc, size, false);
        
        // 返回默认值，等待推断完成后重试
        *pval = 0;
        return MEMTX_OK;
    }
    
    // 调用原始处理函数
    if (original_memory_dispatch_read) {
        return original_memory_dispatch_read(mr, addr, pval, op, attrs);
    }
    
    return MEMTX_DECODE_ERROR;
}

// MMIO写入错误处理
static MemTxResult mmio_write_fault_handler(MemoryRegion *mr, hwaddr addr,
                                           uint64_t val, MemOp op,
                                           MemTxAttrs attrs)
{
    uint64_t size = memop_size(op);
    
    // 检查是否为未映射的MMIO访问
    if (!is_address_mapped(addr, size)) {
        CPUState *cpu = current_cpu;
        uint64_t pc = 0;
        
        if (cpu) {
            CPUClass *cc = CPU_GET_CLASS(cpu);
            if (cc->get_pc) {
                pc = cc->get_pc(cpu);
            }
        }
        
        qemu_log("MMIO Write Fault: addr=0x%lx, val=0x%lx, size=%lu, PC=0x%lx\n", 
                addr, val, size, pc);
        
        // 触发推断处理
        inference_handle_mmio_fault(addr, pc, size, true);
        
        // 返回成功，等待推断完成后重试
        return MEMTX_OK;
    }
    
    // 调用原始处理函数
    if (original_memory_dispatch_write) {
        return original_memory_dispatch_write(mr, addr, val, op, attrs);
    }
    
    return MEMTX_DECODE_ERROR;
}

// 启用错误处理器
void enable_mmio_fault_handler(void)
{
    if (fault_handler_enabled) {
        return;
    }
    
    qemu_log("Enabling MMIO fault handler\n");
    
    // 这里需要hook QEMU的内存分发机制
    // 注意：这是一个简化的示例，实际实现可能需要更复杂的hook机制
    fault_handler_enabled = true;
}

// 禁用错误处理器
void disable_mmio_fault_handler(void)
{
    if (!fault_handler_enabled) {
        return;
    }
    
    qemu_log("Disabling MMIO fault handler\n");
    fault_handler_enabled = false;
} 
