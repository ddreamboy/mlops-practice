from clearml import Task

if __name__ == "__main__":
    task = Task.init(project_name="test", task_name="test-task")
    task.execute_remotely(queue_name="course-queue")