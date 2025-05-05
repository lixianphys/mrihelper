# mrimaster/services/task_store.py
import redis
import json
from datetime import datetime
from typing import Optional, Dict, List



class TaskHandler:
    def __init__(self, redis_host='localhost', redis_port=6379, redis_db=0):
        self.redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True  # Automatically decode Redis responses to strings
        )
        
    def _get_key(self, task_id: str) -> str:
        """Generate Redis key for a task"""
        return f"task:{task_id}"
        
    def create_task(self, task_id: str, image_id: str) -> bool:
        """Create a new task entry"""
        try:
            task_data = {
                "status": "pending",
                "image_id": image_id,
                "created_at": datetime.now().isoformat()
            }
            
            # Store in Redis with 24-hour expiry
            self.redis.setex(
                self._get_key(task_id),
                24 * 60 * 60,  # 24 hours in seconds
                json.dumps(task_data)
            )
            return True
        except Exception:
            return False
        
    def get_task(self, task_id: str) -> Optional[Dict]:
        """Get task data"""
        data = self.redis.get(self._get_key(task_id))
        return json.loads(data) if data else None
    
    def get_all_tasks(self) -> List[Dict]:
        """Get all task data"""
        tasks = []
        for key in self.redis.keys("*"):
            task_data = self.redis.get(key)
            tasks.append(json.loads(task_data))
        return tasks
    
    def update_task(self, task_id: str, updates: Dict) -> bool:
        """Update task data"""
        try:
            key = self._get_key(task_id)
            current_data = self.redis.get(key)
            
            if not current_data:
                return False
                
            task_data = json.loads(current_data)
            task_data.update(updates)
            task_data["updated_at"] = datetime.now().isoformat()
            
            # Update in Redis, maintaining the existing TTL
            ttl = self.redis.ttl(key)
            self.redis.setex(key, ttl if ttl > 0 else 24 * 60 * 60, json.dumps(task_data))
            return True
        except Exception:
            return False