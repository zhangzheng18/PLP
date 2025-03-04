# state_save.py
import os
import time
import uuid
import hashlib
import json
import threading
import multiprocessing
from datetime import datetime
from functools import wraps
from collections import OrderedDict

class StateSnapshotSystem:
    """
    高可靠状态快照管理系统
    支持原子操作、版本控制和自动垃圾回收
    """
    def __init__(self, 
                 snapshot_dir="snapshots",
                 max_versions=5,
                 retention_days=7):
        # 配置参数
        self.snapshot_dir = os.path.abspath(snapshot_dir)
        self.max_versions = max_versions
        self.retention_days = retention_days
        
        # 并发控制
        self.lock = multiprocessing.RLock()
        self.shared_cache = multiprocessing.Manager().dict()
        
        # 初始化目录
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        # 缓存系统
        self.cache = OrderedDict()
        self.cache_size = 10
        self.performance_stats = {
            'snapshot_time': [],
            'restore_time': [],
            'cache_hits': 0
        }
        self.initial_snapshot = None

        # 后台清理线程
        self._start_cleaner_thread()

    def _start_cleaner_thread(self):
        """启动自动清理线程"""
        self.cleaner_thread = threading.Thread(
            target=self._auto_cleanup,
            daemon=True
        )
        self.cleaner_thread.start()

    @staticmethod
    def handle_errors(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except FileNotFoundError as e:
                print(f"File operation failed: {str(e)}")
                raise
            except json.JSONDecodeError:
                print("Metadata corruption detected")
                raise
            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                raise
        return wrapper

    @handle_errors
    def create_snapshot(self, gdb_process):
        """创建系统快照"""
        start_time = time.time()
        with self.lock:
            snapshot_id = self._generate_snapshot_id()
            snapshot_path = self._get_snapshot_path(snapshot_id)
            
            try:
                # 执行GDB保存操作
                self._execute_gdb_save(gdb_process, snapshot_path)
                
                # 读取并验证数据
                data = self._read_and_validate(snapshot_path)
                checksum = self._calculate_checksum(data)
                
                # 更新缓存和元数据
                self._update_cache(snapshot_id, data, checksum)
                self._save_metadata(snapshot_id, {
                    'timestamp': time.time(),
                    'size': len(data),
                    'checksum': checksum
                })
                
                # 记录初始快照
                if not self.initial_snapshot:
                    self.initial_snapshot = snapshot_id
                
                print(f"Snapshot {snapshot_id} created")
                return snapshot_id, checksum
            except Exception as e:
                self._cleanup_failed_snapshot(snapshot_path)
                raise
            finally:
                self._record_performance('snapshot_time', start_time)

    @handle_errors
    def restore_snapshot(self, gdb_process, snapshot_id):
        """恢复系统状态"""
        start_time = time.time()
        with self.lock:
            try:
                data, checksum = self._get_snapshot_data(snapshot_id)
                self._execute_gdb_restore(gdb_process, data)
                self._verify_checksum(data, checksum)
                print(f"Snapshot {snapshot_id} restored")
                return checksum
            finally:
                self._record_performance('restore_time', start_time)

    def get_snapshot_list(self):
        """获取所有可用快照ID"""
        with self.lock:
            cache_keys = list(self.cache.keys())
            disk_snapshots = [
                f[9:-6] for f in os.listdir(self.snapshot_dir)
                if f.startswith("snapshot_") and f.endswith(".state")
            ]
            return list(set(cache_keys + disk_snapshots))

    def get_snapshot_info(self, snapshot_id):
        """获取快照详细信息"""
        meta_path = os.path.join(self.snapshot_dir, f"{snapshot_id}.meta")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                return json.load(f)
        return None

    # 私有方法 -------------------------------------------------
    def _generate_snapshot_id(self):
        return f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"

    def _get_snapshot_path(self, snapshot_id):
        return os.path.join(self.snapshot_dir, f"snapshot_{snapshot_id}.state")

    def _execute_gdb_save(self, gdb_process, path):
        gdb_process.sendline(f"save state {path}")
        gdb_process.expect_exact("(gdb)", timeout=10)

    def _read_and_validate(self, path):
        with open(path, 'rb') as f:
            data = f.read()
        if len(data) < 1024:  # 最小数据校验
            raise ValueError("Invalid snapshot data")
        return data

    def _update_cache(self, snapshot_id, data, checksum):
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)
        self.cache[snapshot_id] = (data, checksum)
        self.cache.move_to_end(snapshot_id)

    def _save_metadata(self, snapshot_id, meta):
        with open(os.path.join(self.snapshot_dir, f"{snapshot_id}.meta"), 'w') as f:
            json.dump(meta, f)

    def _cleanup_failed_snapshot(self, path):
        if os.path.exists(path):
            os.remove(path)

    def _get_snapshot_data(self, snapshot_id):
        if snapshot_id in self.cache:
            self.performance_stats['cache_hits'] += 1
            return self.cache[snapshot_id]
        
        path = self._get_snapshot_path(snapshot_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Snapshot {snapshot_id} not found")
        
        with open(path, 'rb') as f:
            data = f.read()
        return data, self._calculate_checksum(data)

    def _execute_gdb_restore(self, gdb_process, data):
        tmp_path = f"/tmp/{uuid.uuid4()}.state"
        with open(tmp_path, 'wb') as f:
            f.write(data)
        gdb_process.sendline(f"source {tmp_path}")
        gdb_process.expect_exact("(gdb)", timeout=15)
        os.remove(tmp_path)

    def _verify_checksum(self, data, expected):
        actual = self._calculate_checksum(data)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch: {actual} vs {expected}")

    def _calculate_checksum(self, data):
        return hashlib.sha256(data).hexdigest()

    def _record_performance(self, metric_type, start_time):
        duration = time.time() - start_time
        self.performance_stats[metric_type].append(duration)

    def _auto_cleanup(self):
        """自动清理策略"""
        while True:
            try:
                now = time.time()
                snapshots = []
                
                # 扫描有效快照
                for sid in self.get_snapshot_list():
                    meta = self.get_snapshot_info(sid)
                    if meta:
                        snapshots.append((meta['timestamp'], sid))
                
                # 清理策略
                snapshots.sort(reverse=True)
                keep_count = min(self.max_versions, len(snapshots))
                expired = [
                    sid for ts, sid in snapshots[keep_count:]
                    if (now - ts) > self.retention_days * 86400
                ]
                
                for sid in expired:
                    self._delete_snapshot(sid)
                
                time.sleep(3600)
            except Exception as e:
                print(f"Cleanup error: {str(e)}")

    def _delete_snapshot(self, sid):
        """安全删除快照"""
        with self.lock:
            try:
                # 删除状态文件
                state_path = self._get_snapshot_path(sid)
                if os.path.exists(state_path):
                    os.remove(state_path)
                
                # 删除元数据
                meta_path = os.path.join(self.snapshot_dir, f"{sid}.meta")
                if os.path.exists(meta_path):
                    os.remove(meta_path)
                
                # 清理缓存
                if sid in self.cache:
                    del self.cache[sid]
            except Exception as e:
                print(f"Delete failed for {sid}: {str(e)}")

    def get_performance_metrics(self):
        """获取性能统计"""
        return {
            'total_snapshots': len(self.performance_stats['snapshot_time']),
            'average_snapshot_time': sum(self.performance_stats['snapshot_time'])/len(self.performance_stats['snapshot_time']) if self.performance_stats['snapshot_time'] else 0,
            'cache_hit_rate': self.performance_stats['cache_hits'] / len(self.performance_stats['restore_time']) if len(self.performance_stats['restore_time']) > 0 else 0
        }

# 单例实例和模块接口 --------------------------------------------
_system_instance = StateSnapshotSystem()

def save_snapshot(gdb):
    return _system_instance.create_snapshot(gdb)

def restore_snapshot(gdb, sid):
    return _system_instance.restore_snapshot(gdb, sid)

def get_snapshot_list():
    return _system_instance.get_snapshot_list()

def get_snapshot_info(sid):
    return _system_instance.get_snapshot_info(sid)

def get_performance_metrics():
    return _system_instance.get_performance_metrics()