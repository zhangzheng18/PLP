#include "qemu/osdep.h"
#include "qemu/log.h"
#include "qemu/notify.h"
#include "qom/object.h"
#include "hw/qdev-core.h"
#include "hw/qdev-properties.h"
#include "hw/sysbus.h"
#include "exec/memory.h"
#include "hw/pci/pci.h"
#include "hw/pci/pci_device.h"
#include "qapi/qmp/qdict.h"
#include "qapi/qmp/qjson.h"
#include "qemu/cutils.h"
#include "sysemu/sysemu.h"
#include "qapi/error.h"
#include "qapi/qmp/qobject.h"
#include "glib-compat.h"
#include <stdio.h>
#include <sys/stat.h>
#include <errno.h>
#include <unistd.h>

#define OUTPUT_PATH "log/device_map.json"

static Notifier machine_init_done_notifier;

static void ensure_log_directory(void) {
    struct stat st = {0};
    if (stat("log", &st) == -1) {
        if (mkdir("log", 0755) == -1) {
            fprintf(stderr, "Warning: Failed to create log directory: %s\n", strerror(errno));
        }
    }
}

static int qdev_get_gpio_in_count_safe(DeviceState *dev) {
    if (!dev) return 0;
    Object *obj = OBJECT(dev);
    if (!obj || !object_property_find(obj, "num-gpio-in")) {
        return 0;
    }
    Error *err = NULL;
    int count = object_property_get_int(obj, "num-gpio-in", &err);
    if (err) {
        error_free(err);
        return 0;
    }
    return count;
}

static bool is_device_interesting(const char *type_name) {
    // Filter for interesting device types - peripherals, not internal QEMU objects
    if (!type_name) return false;
    
    // Skip QEMU internal objects
    if (g_str_has_prefix(type_name, "memory-backend-") ||
        g_str_has_prefix(type_name, "rng-") ||
        g_str_has_prefix(type_name, "secret") ||
        g_str_has_prefix(type_name, "authz-") ||
        g_str_has_prefix(type_name, "iothread") ||
        g_str_has_prefix(type_name, "migration") ||
        g_str_equal(type_name, "container") ||
        g_str_equal(type_name, "qemu:memory-region")) {
        return false;
    }
    
    // Include interesting peripheral devices
    if (g_str_has_suffix(type_name, "-device") ||
        g_str_has_suffix(type_name, "-controller") ||
        strstr(type_name, "uart") != NULL ||
        strstr(type_name, "serial") != NULL ||
        strstr(type_name, "timer") != NULL ||
        strstr(type_name, "gpio") != NULL ||
        strstr(type_name, "rtc") != NULL ||
        strstr(type_name, "flash") != NULL ||
        strstr(type_name, "virtio") != NULL ||
        strstr(type_name, "pci") != NULL ||
        strstr(type_name, "PCI") != NULL ||
        strstr(type_name, "pl011") != NULL ||
        strstr(type_name, "pl061") != NULL ||
        strstr(type_name, "pl080") != NULL ||
        strstr(type_name, "arm") != NULL ||
        strstr(type_name, "cfi") != NULL ||
        strstr(type_name, "platform") != NULL ||
        strstr(type_name, "mmio") != NULL) {
        return true;
    }
    
    return false;
}

static uint64_t try_get_device_address(Object *obj, const char *dev_type) {
    Error *err = NULL;
    uint64_t addr = 0;
    
    // Try multiple common address property names
    const char *addr_props[] = {"mmio", "base", "addr", "address", "reg", NULL};
    
    for (int i = 0; addr_props[i]; i++) {
        if (object_property_find(obj, addr_props[i])) {
            addr = object_property_get_uint(obj, addr_props[i], &err);
            if (!err && addr != 0 && addr != (uint64_t)-1) {
                printf("  Found address 0x%lx via property '%s'\n", addr, addr_props[i]);
                return addr;
            }
            if (err) {
                error_free(err);
                err = NULL;
            }
        }
    }
    
    // For specific device types, try known addresses (ARM virt platform)
    if (strstr(dev_type, "pl011")) {
        // ARM virt platform typically puts first UART at 0x9000000
        printf("  Using default pl011 address 0x9000000\n");
        return 0x9000000;
    } else if (strstr(dev_type, "pl061")) {
        // GPIO typically at 0x9030000
        printf("  Using default pl061 address 0x9030000\n");  
        return 0x9030000;
    } else if (strstr(dev_type, "pl080")) {
        // DMA controller typically at 0x9000000-0x900ffff range
        printf("  Using default pl080 address 0x9010000\n");
        return 0x9010000;
    } else if (strstr(dev_type, "rtc")) {
        // RTC typically at 0x9010000
        printf("  Using default rtc address 0x9010000\n");
        return 0x9010000;
    }
    
    return 0;
}

static void collect_mmio_info(Object *obj, QDict *ddev) {
    // Try to get MMIO information safely
    QDict *mmio_regions = qdict_new();
    bool has_mmio = false;
    const char *dev_type = object_get_typename(obj);
    
    // For SysBus devices, try to get memory regions
    if (object_dynamic_cast(obj, "sys-bus-device")) {
        SysBusDevice *sbd = SYS_BUS_DEVICE(obj);
        if (sbd) {
            for (int i = 0; i < QDEV_MAX_MMIO; i++) {
                if (sbd->mmio[i].memory) {
                    MemoryRegion *mr = sbd->mmio[i].memory;
                    if (mr && memory_region_size(mr) > 0) {
                        QDict *region_info = qdict_new();
                        qdict_put_int(region_info, "size", memory_region_size(mr));
                        qdict_put_str(region_info, "name", memory_region_name(mr) ?: "unnamed");
                        
                        uint64_t base_addr = 0;
                        // Try to get the address if mapped
                        if (sbd->mmio[i].addr != (hwaddr)-1) {
                            base_addr = sbd->mmio[i].addr;
                            printf("  Found mapped address 0x%lx for %s\n", base_addr, memory_region_name(mr));
                        } else {
                            // Try to get address through other means
                            base_addr = try_get_device_address(obj, dev_type);
                        }
                        
                        if (base_addr != 0) {
                            qdict_put_int(region_info, "base", base_addr);
                        }
                        
                        char key[16];
                        snprintf(key, sizeof(key), "mmio_%d", i);
                        qdict_put(mmio_regions, key, region_info);
                        has_mmio = true;
                    }
                }
            }
        }
    }
    
    // If no MMIO regions found through SysBus, try direct property access
    if (!has_mmio) {
        uint64_t addr = try_get_device_address(obj, dev_type);
        if (addr != 0) {
            QDict *region_info = qdict_new();
            qdict_put_int(region_info, "base", addr);
            qdict_put_str(region_info, "name", dev_type);
            // Try to guess size for known devices
            if (strstr(dev_type, "pl011")) {
                qdict_put_int(region_info, "size", 4096);
            } else if (strstr(dev_type, "pl061")) {
                qdict_put_int(region_info, "size", 4096);
            } else {
                qdict_put_int(region_info, "size", 4096); // Default 4KB
            }
            qdict_put(mmio_regions, "mmio_0", region_info);
            has_mmio = true;
        }
    }
    
    if (has_mmio) {
        qdict_put_obj(ddev, "mmio_regions", QOBJECT(mmio_regions));
    } else {
        qobject_unref(mmio_regions);
    }
}

static void collect_irq_info(Object *obj, DeviceState *dev, QDict *ddev) {
    // IRQ information
    int num_irq = qdev_get_gpio_in_count_safe(dev);
    if (num_irq > 0) {
        qdict_put_int(ddev, "num_irqs", num_irq);
    }
    
    // For SysBus devices, also try to get IRQ numbers using proper API
    if (object_dynamic_cast(obj, "sys-bus-device")) {
        SysBusDevice *sbd = SYS_BUS_DEVICE(obj);
        if (sbd) {
            QDict *irq_info = qdict_new();
            bool has_irqs = false;
            
            // Check up to 32 possible IRQ lines (reasonable upper limit)
            for (int i = 0; i < 32; i++) {
                if (sysbus_has_irq(sbd, i)) {
                    char key[16];
                    snprintf(key, sizeof(key), "irq_%d", i);
                    qdict_put_bool(irq_info, key, true);
                    has_irqs = true;
                }
            }
            
            if (has_irqs) {
                qdict_put_obj(ddev, "irq_lines", QOBJECT(irq_info));
            } else {
                qobject_unref(irq_info);
            }
        }
    }
}

static int dump_device(Object *obj, void *opaque) {
    QDict *device_list = (QDict *)opaque;

    // First check if it's a device
    if (!object_dynamic_cast(obj, "device")) {
        return 0;
    }

    DeviceState *dev = DEVICE(obj);
    if (!dev || !dev->realized) {
        return 0;
    }
    
    const char *name = object_get_typename(obj);
    char *path = object_get_canonical_path(obj);
    if (!name || !path) {
        g_free(path);
        return 0;
    }

    // Filter for interesting devices only
    if (!is_device_interesting(name)) {
        g_free(path);
        return 0;
    }

    printf("Processing device: %s at %s\n", name, path);

    QDict *ddev = qdict_new();
    qdict_put_str(ddev, "type", name);
    qdict_put_str(ddev, "path", path);

    // compatible - be very careful with property access
    if (object_property_find(obj, "compatible")) {
        Error *err = NULL;
        char *compat = object_property_get_str(obj, "compatible", &err);
        if (compat && !err) {
            qdict_put_str(ddev, "compatible", compat);
            g_free(compat);
        }
        if (err) {
            error_free(err);
        }
    }

    // Collect MMIO information
    collect_mmio_info(obj, ddev);
    
    // Collect IRQ information
    collect_irq_info(obj, dev, ddev);

    // PCI device info - much more conservative approach
    // Only try PCI operations if we're really sure it's a PCI device
    if (strstr(name, "pci") != NULL || strstr(name, "PCI") != NULL) {
        // Double-check with a safer cast approach
        Object *pci_obj = object_dynamic_cast(obj, "pci-device");
        if (pci_obj) {
            // Additional safety check - verify the object has the expected structure
            if (object_property_find(obj, "addr") && 
                object_property_find(obj, "vendor-id")) {
                Error *err = NULL;
                uint32_t vendor_id = object_property_get_uint(obj, "vendor-id", &err);
                if (!err) {
                    QDict *pci_info = qdict_new();
                    qdict_put_int(pci_info, "vendor_id", vendor_id);
                    
                    uint32_t device_id = object_property_get_uint(obj, "device-id", &err);
                    if (!err) {
                        qdict_put_int(pci_info, "device_id", device_id);
                    }
                    if (err) {
                        error_free(err);
                        err = NULL;
                    }
                    
                    // Try to get PCI address
                    uint32_t addr = object_property_get_uint(obj, "addr", &err);
                    if (!err) {
                        qdict_put_int(pci_info, "pci_addr", addr);
                    }
                    if (err) {
                        error_free(err);
                    }
                    
                    qdict_put_obj(ddev, "pci_info", QOBJECT(pci_info));
                }
                if (err) {
                    error_free(err);
                }
            }
        }
    }

    // 使用递增数字作为键，避免路径冲突
    static int device_counter = 0;
    char device_key[32];
    snprintf(device_key, sizeof(device_key), "device_%d", device_counter++);
    qdict_put_obj(device_list, device_key, QOBJECT(ddev));
    
    g_free(path);
    return 0;
}

static void dump_all_devices_json(QDict *device_list) {
    printf("Starting device enumeration...\n");
    object_child_foreach_recursive(object_get_root(), dump_device, device_list);
    printf("Device enumeration completed.\n");
}

static void save_json_to_file(QDict *device_list) {
    ensure_log_directory();
    
    GString *json = qobject_to_json_pretty(QOBJECT(device_list), true);
    FILE *fp = fopen(OUTPUT_PATH, "w");
    if (!fp) {
        perror("fopen device_map.json");
        g_string_free(json, true);
        return;
    }
    fwrite(json->str, 1, json->len, fp);
    fclose(fp);
    g_string_free(json, true);
    printf("===> Device info saved to %s\n", OUTPUT_PATH);
}

static void machine_init_done_cb(Notifier *notifier, void *data) {
    printf("===> Device Dump Start <===\n");
    QDict *device_list = qdict_new();
    dump_all_devices_json(device_list);
    save_json_to_file(device_list);
    qobject_unref(device_list);
    printf("===> Device Dump End <===\n");
}

static void device_dump_init(void) {
    // Register a machine init done notifier to safely dump devices
    // after QEMU is fully initialized
    machine_init_done_notifier.notify = machine_init_done_cb;
    qemu_add_machine_init_done_notifier(&machine_init_done_notifier);
}

type_init(device_dump_init);
