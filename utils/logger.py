import time


class AgentLogger:

    def __init__(self, agent_name):

        self.agent_name = agent_name
        self.start = None

    def begin(self, task):

        self.start = time.time()

        print("\n" + "=" * 70)
        print(f"🤖 {self.agent_name}")
        print("=" * 70)
        print(f"Task : {task}")
        print()

    def end(self, output=""):

        elapsed = time.time() - self.start

        print()
        print("-" * 70)

        if output:
            print(output)

        print()
        print(f"Completed in : {elapsed:.2f} sec")
        print("=" * 70)