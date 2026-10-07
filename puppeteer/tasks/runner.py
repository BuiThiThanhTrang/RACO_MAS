from agent.register.register import agent_global_registry
from inference.reasoning.reasoning import GraphReasoning
from inference.graph.agent_graph import AgentGraph

class BenchmarkRunner:
    def __init__(self, personas_path, global_config):
        self.personas_path = personas_path
        self.global_config = global_config
        self.max_step_num = self.global_config.get('graph').get('max_step_num')
        self.save_state = False
        self.policy = None
        # Keep agent identities and the policy action dimension stable throughout
        # a dataset run. Re-registering per row accumulated duplicate agents.
        agent_global_registry.register_all_agents(self.personas_path)

    def setup_reasoning(self, data_item):
        agent_global_registry.reset_all_agents()
        graph = AgentGraph()
        return GraphReasoning(data_item, graph), graph

    def run_reasoning(self, data_item):
        reasoning, _ = self.setup_reasoning(data_item)
        self.policy = reasoning.policy
        reasoning.start(self.save_state if self.save_state else None)
        self.save_state = False
        
        final_ans, _ = reasoning.n_step(self.max_step_num)

        reasoning.visualize_path()
        reasoning.visualize_graph()

        return final_ans

    def save_resume_checkpoint(self, completed_rows, result_last_id=None):
        if self.policy is None:
            raise RuntimeError("Policy is not initialized; cannot save a checkpoint")
        checkpoint = self.policy.save_resume_checkpoint(
            completed_rows=completed_rows,
            result_last_id=result_last_id,
        )
        if checkpoint is None:
            raise RuntimeError("Failed to save the resume checkpoint")
        return checkpoint

    def loaded_checkpoint_rows(self):
        if self.policy is None:
            return None
        return self.policy.loaded_checkpoint_metadata.get("completed_rows")
