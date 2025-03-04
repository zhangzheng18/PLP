class SnapshotConfig:
    SNAPSHOT_DIR = "sim_snapshots"
    MAX_VERSIONS = 10
    RETENTION_DAYS = 3
    CACHE_SIZE = 15  # 增大缓存容量
    AUTO_CLEAN_INTERVAL = 3600  # 清理间隔(秒)