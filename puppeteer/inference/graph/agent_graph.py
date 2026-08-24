import json
import yaml
from inference.base.graph import Graph
import logging
main_logger = logging.getLogger('global') 

class AgentGraph(Graph):
    def __init__(self, registry, profile_store=None, allowed_tools=()):
        super().__init__()
        if registry is None:
            raise ValueError("AgentGraph requires an explicit run-scoped registry")
        self.registry = registry
        self.profile_store = profile_store
        self.allowed_tools = frozenset(allowed_tools)
        self._nodes_num = self.registry.agent_num
        self._edges_num = 0
        for agent in self.registry.ordered_agents:
            self._add_node(agent)
        print("-"*10+"\033[31mAgent Graph Initialized\033[0m"+"-"*10)

    @property
    def hash_nodes(self):
        return [node.hash for node in self._nodes]
    
    @property
    def role_nodes(self):
        return [node.role for node in self._nodes]
    
    def get_agent_from_index(self, index):
        return self._nodes[index]
    
    def get_agent_from_role(self, role):
        for agent in self._nodes:
            if agent.role == role:
                return agent
        return None
    
    def get_agent_from_hash(self, hash):
        for agent in self._nodes:
            if agent.hash == hash:
                return agent
        return None
    
    def get_agent_dialog_history(self, agent_role_list: list, **kwargs):
        """get agent dialog history
        
        Keyword arguments:
        idx -- agent idx
        Return: corresponding agent dialog history. If idx is illegal, return []
        """
        question = kwargs.get("question", None)
        history = []
        for role in agent_role_list:
            agent = self.get_agent_from_role(role)
            for h in agent.simplified_dialog_history:
                history.append(h)
        if len(agent_role_list) == 0 and question is not None:
            history = [{'role': 'system', 'content': 'You are an assistant. Your task is to {}'.format(question)}]
        assert len(history)!=0, "Dialog history can not be empty"
        return history    
    
    def is_agent_available(self, agent):
        return bool(
            agent.spec.available
            and all(tool in self.allowed_tools for tool in agent.role_card.tools)
        )

    @property
    def availability_mask(self):
        return tuple(self.is_agent_available(agent) for agent in self._nodes)

    @property
    def agent_prompt(self):
        return "\n".join(
            f"Candidate {index}: {json.dumps(view, ensure_ascii=False, sort_keys=True)}"
            for index, view in enumerate(self.public_agent_views())
        )

    def public_agent_views(self):
        if self.profile_store is None:
            return tuple(
                agent.role_card.to_dict() | {"available": self.is_agent_available(agent)}
                for agent in self._nodes
            )
        specs = tuple(agent.spec for agent in self._nodes)
        views = self.profile_store.public_views(specs)
        return tuple(
            view.to_dict() | {"available": self.is_agent_available(agent)}
            for view, agent in zip(views, self._nodes)
        )
    
    @property
    def terminator_agent_index(self):
        for agent in self._nodes:
            if "terminate" in agent.actions:
                return agent.index
        return None
    
    @property
    def search_agent_indices(self):
        indices = []
        for agent in self._nodes:
            if any(tool in agent.tools for tool in ("access_website", "search_bing", "search_arxiv")):
                indices.append(agent.index)
        return indices
    
    def agent_list(self):
        return self.agent_prompt

    def reset_episode_state(self):
        self._edges.clear()
        self._edges_num = 0
    
    def visualize(self, path="agent_graph.html"):
        try:
            import networkx as nx
            from pyvis.network import Network
            import seaborn as sns
        except ImportError as e:
            print(f"Skipping agent graph visualization because dependencies are missing: {e}")
            return

        def generate_color_map(node_ids):
            color_palette = sns.color_palette("husl", len(node_ids)).as_hex()
            color_map = {node_id: color_palette[i % len(color_palette)] for i, node_id in enumerate(node_ids)}
            return color_map
        node_color_map = generate_color_map(self.hash_nodes)
        edge_color_map = generate_color_map([edge.index for edge in self._edges])
        
        G = nx.MultiDiGraph()
        edge_labels = {}
        for node in self._nodes:
            G.add_node(node.index, label=f"{node.role}\nbase model: {node.model}\nindex: {node.index}",color = node_color_map[node.hash])
        
        for edge in self._edges:
            G.add_edge(edge.v.index, edge.u.index, color = edge_color_map[edge.index])
            edge_labels[(edge.v.index, edge.u.index)] = f"Reasoning..."
        
        net = Network(notebook=True, height="750px", width="100%", bgcolor="#FFFFFF", font_color="black", directed=True)
        net.from_nx(G)
        net.show(path)
    
    @property
    def num(self):
        return self._nodes_num
    
    def add_agent(self):
        pass
    def delete_agent(self):
        pass
