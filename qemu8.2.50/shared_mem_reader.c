#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <stdint.h>
#include <time.h>

#define SHARED_MEM_SIZE 4096
#define MAX_LOG_ENTRIES 100

// 与mmio_proxy.h中定义相同的结构
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

static void print_timestamp(uint64_t timestamp_us)
{
    time_t sec = timestamp_us / 1000000;
    uint64_t usec = timestamp_us % 1000000;
    struct tm *tm_info = localtime(&sec);
    printf("%02d:%02d:%02d.%06lu", 
           tm_info->tm_hour, tm_info->tm_min, tm_info->tm_sec, usec);
}

static void print_entry(const StateLogEntry *entry, int index)
{
    printf("\n=== Entry %d ===\n", index);
    printf("Timestamp: ");
    print_timestamp(entry->timestamp);
    printf("\n");
    
    printf("CPU ID: %u\n", entry->cpu_id);
    if (entry->irq_num != (uint32_t)-1) {
        printf("IRQ: %u\n", entry->irq_num);
    }
    printf("PC: 0x%016lx\n", entry->pc);
    printf("SP: 0x%016lx\n", entry->sp);
    
    if (entry->mmio_addr != 0) {
        printf("MMIO %s: addr=0x%016lx, val=0x%016lx, size=%u\n",
               entry->is_write ? "WRITE" : "READ",
               entry->mmio_addr, entry->mmio_val, entry->mmio_size);
    }
    
    // 打印部分寄存器状态
    printf("Registers (X0-X7):\n");
    for (int i = 0; i < 8; i++) {
        printf("  X%-2d = 0x%016lx", i, entry->xregs[i]);
        if ((i + 1) % 2 == 0) printf("\n");
    }
    if (8 % 2 != 0) printf("\n");
    
    // 打印外设寄存器状态（前32字节）
    printf("Device Registers (first 32 bytes):\n");
    for (int i = 0; i < 32; i += 8) {
        uint64_t val = 0;
        memcpy(&val, &entry->mmio_regs[i], 8);
        printf("  [0x%02x] = 0x%016lx\n", i, val);
    }
}

static void monitor_shared_memory(const char *shm_name)
{
    int fd = shm_open(shm_name, O_RDONLY, 0);
    if (fd == -1) {
        fprintf(stderr, "Failed to open shared memory '%s': %s\n", 
                shm_name, strerror(errno));
        return;
    }
    
    SharedMemoryLog *log = mmap(NULL, SHARED_MEM_SIZE, PROT_READ, 
                               MAP_SHARED, fd, 0);
    if (log == MAP_FAILED) {
        fprintf(stderr, "Failed to map shared memory: %s\n", strerror(errno));
        close(fd);
        return;
    }
    
    printf("Monitoring shared memory: %s\n", shm_name);
    printf("Press Ctrl+C to exit\n\n");
    
    uint32_t last_count = 0;
    
    while (1) {
        if (log->entry_count > last_count) {
            // 有新的条目
            uint32_t start_index = last_count;
            uint32_t end_index = log->entry_count;
            
            for (uint32_t i = start_index; i < end_index; i++) {
                uint32_t real_index = i % MAX_LOG_ENTRIES;
                print_entry(&log->entries[real_index], i);
            }
            
            last_count = log->entry_count;
        }
        
        usleep(100000); // 100ms
    }
    
    munmap(log, SHARED_MEM_SIZE);
    close(fd);
}

static void dump_all_entries(const char *shm_name)
{
    int fd = shm_open(shm_name, O_RDONLY, 0);
    if (fd == -1) {
        fprintf(stderr, "Failed to open shared memory '%s': %s\n", 
                shm_name, strerror(errno));
        return;
    }
    
    SharedMemoryLog *log = mmap(NULL, SHARED_MEM_SIZE, PROT_READ, 
                               MAP_SHARED, fd, 0);
    if (log == MAP_FAILED) {
        fprintf(stderr, "Failed to map shared memory: %s\n", strerror(errno));
        close(fd);
        return;
    }
    
    printf("Shared memory dump: %s\n", shm_name);
    printf("Total entries: %u, Write index: %u\n\n", 
           log->entry_count, log->write_index);
    
    uint32_t count = log->entry_count;
    if (count > MAX_LOG_ENTRIES) {
        count = MAX_LOG_ENTRIES;
    }
    
    for (uint32_t i = 0; i < count; i++) {
        print_entry(&log->entries[i], i);
    }
    
    munmap(log, SHARED_MEM_SIZE);
    close(fd);
}

int main(int argc, char *argv[])
{
    const char *shm_name = "/mmio_proxy_shared";
    int monitor_mode = 0;  // 用int代替bool
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--monitor") == 0) {
            monitor_mode = 1;  // 用1代替true
        } else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--name") == 0) {
            if (i + 1 < argc) {
                shm_name = argv[++i];
            } else {
                fprintf(stderr, "Error: -n requires a name argument\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  -m, --monitor     Monitor mode (continuous)\n");
            printf("  -n, --name NAME   Shared memory name (default: %s)\n", shm_name);
            printf("  -h, --help        Show this help\n");
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
    }
    
    if (monitor_mode) {
        monitor_shared_memory(shm_name);
    } else {
        dump_all_entries(shm_name);
    }
    
    return 0;
} 
