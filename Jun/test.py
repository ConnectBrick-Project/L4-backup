# workflow_analyzer.py
import json

class WorkflowAnalyzer:
    def __init__(self, workflow_path):
        self.workflow_path = workflow_path
        self.load_workflow()
        
    def load_workflow(self):
        with open(self.workflow_path, 'r') as f:
            self.workflow = json.load(f)
            
    def get_node_types(self):
        """워크플로우에서 사용된 모든 노드 타입 추출"""
        node_types = {}
        for node_id, node_data in self.workflow.items():
            node_type = node_data.get('class_type')
            if node_type:
                if node_type not in node_types:
                    node_types[node_type] = []
                node_types[node_type].append(node_id)
        return node_types
    
    def find_input_nodes(self):
        """사용자 입력이 필요한 노드 찾기"""
        input_nodes = []
        for node_id, node_data in self.workflow.items():
            inputs = node_data.get('inputs', {})
            if any(isinstance(v, (str, int, float)) for v in inputs.values()):
                input_nodes.append({
                    'node_id': node_id,
                    'class_type': node_data.get('class_type'),
                    'inputs': inputs
                })
        return input_nodes

# 실행 코드
if __name__ == "__main__":
    # workflow.json 파일의 실제 경로를 지정해주세요
    analyzer = WorkflowAnalyzer("test.json")
    
    # 노드 타입 분석
    print("\n=== Node Types ===")
    node_types = analyzer.get_node_types()
    for type_name, nodes in node_types.items():
        print(f"\n{type_name}:")
        for node_id in nodes:
            print(f"  - Node ID: {node_id}")
    
    # 입력 노드 찾기
    print("\n=== Input Nodes ===")
    input_nodes = analyzer.find_input_nodes()
    for node in input_nodes:
        print(f"\nNode ID: {node['node_id']}")
        print(f"Class Type: {node['class_type']}")
        print("Inputs:", node['inputs'])