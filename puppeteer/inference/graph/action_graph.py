from inference.base.graph import Graph
from agent.agent_info.actions import REASONING_ACTION_LIST, TOOL_ACTION_LIST, TERMINATION_ACTION_LIST

class ActionGraph(Graph):
    def __init__(self, allowed_tools=()):
        super().__init__()
        self.REASONING_ACTION_LIST = REASONING_ACTION_LIST
        self.TOOL_ACTION_LIST = TOOL_ACTION_LIST
        self.TERMINATION_ACTION_LIST = TERMINATION_ACTION_LIST
        self.allowed_tools = tuple(allowed_tools)
        enabled_tools = [tool for tool in TOOL_ACTION_LIST if tool in self.allowed_tools]
        self.actions_collection = REASONING_ACTION_LIST + enabled_tools + TERMINATION_ACTION_LIST

    def reset_episode_state(self):
        self._nodes.clear()
        self._edges.clear()
        self._nodes_num = 0
        self._edges_num = 0


    def add_action(self, action_id, action_data, agent_data):
        self._add_node({"id": action_id, "action": action_data, "agent": agent_data})

    def add_dependency(self, from_action_id, to_action_id):
        self._add_edge(from_action_id, to_action_id, len(self._edges))

    def visualize(self, path="action_graph.html"):
        try:
            import networkx as nx
            from pyvis.network import Network
        except ImportError as e:
            print(f"Skipping action graph visualization because dependencies are missing: {e}")
            return

        G = nx.DiGraph()
        nodes_colors = []
        for node in self._nodes:
            G.add_node(node["id"], label=node["action"]["action"]["action"] + "\n" + node["agent"], 
                       status=node["action"]["success"], 
                       color="green" if node["action"]["success"] == "Success" else "red")
            nodes_colors.append("green" if node["action"]["success"] == "Success" else "red")
        for edge in self._edges:
            G.add_edge(edge.u, edge.v)
        net = Network(notebook=True, height="750px", width="100%", bgcolor="#FFFFFF", font_color="black", directed=True)
        net.from_nx(G)
        net.show(path)

    def get_action_data(self, action_id):
        for node in self._nodes:
            if node["id"] == action_id:
                return node
        return None
    
    def get_dependencies(self, action_id):
        return [edge.v for edge in self._edges if edge.u == action_id]
