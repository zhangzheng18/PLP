#include "qemu/osdep.h"
#include "target/arm/cpu.h"
#include "hw/arm/mmio_proxy.h"
#include "hw/arm/gic_arm_hook.h"
#include "qemu/log.h"
#include "cpu.h"
#include <string.h>

void dump_cpu_state_for_irq(int cpu_index, int irq)
{
    CPUState *cs = qemu_get_cpu(cpu_index);
    if (!cs || !object_dynamic_cast(OBJECT(cs), TYPE_ARM_CPU)) {
        qemu_log("IRQ HOOK: No valid ARM CPU found for cpu_index=%d\n", cpu_index);
        return;
    }
    ARMCPU *cpu = ARM_CPU(cs);
    CPUARMState *env = &cpu->env;
    qemu_log("[IRQ HOOK] IRQ=%d, CPU=%d\n", irq, cpu_index);
    for (int i = 0; i < 31; i++) {
        qemu_log("  X%-2d = 0x%016lx\n", i, (unsigned long)env->xregs[i]);
    }
    qemu_log("  SP   = 0x%016lx\n", (unsigned long)env->sp_el[0]);
    qemu_log("  PC   = 0x%016lx\n", (unsigned long)env->pc);
    
    if (global_mmio_proxy) {
        qemu_log("[IRQ HOOK] MMIOProxy REG DUMP:\n");
        for (unsigned long i = 0; i < global_mmio_proxy->size; i += 8) {
            unsigned long regval = 0;
            size_t copy_size = (i + 8 <= global_mmio_proxy->size) ? 8 : (global_mmio_proxy->size - i);
            memcpy(&regval, &global_mmio_proxy->regs[i], copy_size);
            qemu_log("  REG[0x%02lx] = 0x%016lx\n", i, regval);
        }
    }
}

void gic_activate_irq_hook(int irq_num) {
    // 当指定的IRQ被激活时，记录系统状态
    dump_cpu_state_for_irq(0, irq_num);  // 假设使用CPU 0
}


