import argparse
import sys
from pathlib import Path
from typing import Optional

from src.main import TranscriptomeAgent, run_agent
from src.config import config


def format_result(result: dict) -> str:
    """Format the result dictionary into readable text"""
    if not result.get('success'):
        error_msg = result.get('error', result.get('answer', 'Unknown error'))
        return f"❌ Error: {error_msg}"
    
    answer = result.get('answer', {})
    
    if isinstance(answer, str):
        return answer
    
    if isinstance(answer, dict):
        if 'input' in answer and isinstance(answer['input'], dict):
            answer = answer['input']
        
        message = answer.get('message', '')
        if message:
            lines = message.split('\n')
            output = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith('**') and line.endswith('**'):
                    line = f"📌 {line.strip('*')}"
                elif line.startswith('**'):
                    line = f"  {line.replace('**', '')}"
                elif line.startswith('-') or line.startswith('*'):
                    line = f"  {line}"
                elif line.startswith('`'):
                    line = f"  📄 {line.strip('`')}"
                else:
                    line = f"  {line}"
                output.append(line)
            return '\n'.join(output)
    
    return str(answer)


def main():
    parser = argparse.ArgumentParser(
        description="Transcriptome Data Visualization Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with config file (recommended)
  python -m src.cli --request "Create a bar plot for Gene_0001, Gene_0002, Gene_0003"
  
  # Run with explicit parameters
  python -m src.cli --api-key YOUR_KEY --data data/expression.csv --request "Create a heatmap"

  # Interactive mode with config
  python -m src.cli --interactive

  # Specify custom config file
  python -m src.cli --config my_config.yaml --request "Create a volcano plot"
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )

    parser.add_argument(
        '--api-key',
        type=str,
        help='DeepSeek API key (overrides config file)'
    )

    parser.add_argument(
        '--data',
        type=str,
        help='Path to expression data file (CSV, TSV, Excel) (overrides config file)'
    )

    parser.add_argument(
        '--request',
        type=str,
        help='Natural language visualization request'
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Model to use (overrides config file)'
    )

    parser.add_argument(
        '--max-iterations',
        type=int,
        help='Maximum ReAct iterations (overrides config file)'
    )

    parser.add_argument(
        '--show-full-steps',
        action='store_true',
        help='Show full ReAct execution steps without truncation'
    )

    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for saved visualizations (overrides config file)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output (overrides config file)'
    )

    args = parser.parse_args()

    config.reload()

    api_key = args.api_key or config.get_api_key()
    data_path = args.data or config.get_default_data_path()
    model = args.model or config.get_model()
    max_iterations = args.max_iterations or config.get_max_iterations()
    output_dir = args.output_dir or config.get_output_dir()
    verbose = args.verbose if args.verbose is not None else config.get_verbose()

    if not api_key:
        print("Error: API key not found. Please provide --api-key or set it in config.yaml")
        sys.exit(1)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    if args.interactive:
        run_interactive(api_key, data_path, model, output_dir_path, verbose)
    elif args.request:
        if not data_path:
            print("Error: --data is required when --request is provided (or set default in config.yaml)")
            sys.exit(1)

        result = run_agent(
            api_key=api_key,
            user_request=args.request,
            data_path=data_path,
            model=model,
            verbose=verbose,
            output_dir=str(output_dir_path)
        )

        print("\n" + "="*60)
        print("✅ 可视化结果")
        print("="*60)
        print(format_result(result))
        print("="*60)

        if result.get('steps'):
            print("\nReAct Execution Steps:")
            for i, step in enumerate(result['steps']):
                print(f"\nStep {i+1}:")
                if 'thought' in step:
                    thought = step['thought']['content']
                    if args.show_full_steps:
                        print(f"  Thought: {thought}")
                    else:
                        print(f"  Thought: {thought[:100]}...")
                if 'action' in step:
                    action = step['action']
                    if args.show_full_steps:
                        print(f"  Action: {action['tool_name']}")
                        if 'tool_input' in action:
                            print(f"    Input: {action['tool_input']}")
                    else:
                        print(f"  Action: {action['tool_name']}")
                if 'observation' in step:
                    obs = step['observation']
                    if args.show_full_steps:
                        print(f"  Observation: {obs}")
                    else:
                        print(f"  Observation: {'Success' if obs['success'] else obs['error']}")
    else:
        print("Error: Either --request or --interactive must be specified")
        sys.exit(1)


def run_interactive(api_key: str, data_path: Optional[str], model: str, output_dir: Path, verbose: bool):
    print("="*60)
    print("Transcriptome Visualization Agent - Interactive Mode")
    print("="*60)
    print("Type 'quit' or 'exit' to end the session")
    print("Type 'help' for available commands")
    print("="*60)

    agent = TranscriptomeAgent(api_key=api_key, model=model, verbose=verbose, output_dir=str(output_dir))

    if data_path:
        print(f"\nLoading data from: {data_path}")
        result = agent.load_data(data_path)
        if result.get('success'):
            print(f"Data loaded successfully: {result.get('data_shape')}")
            info = agent._get_data_info()
            if info.get('success'):
                print(f"Columns: {info['info']['columns'][:5]}...")
            
            from src.react_agent.core import Thought, Action, Observation, ReActStep
            thought = Thought(content=f"Data has been loaded from {data_path}. Shape: {result.get('data_shape')}. Columns: {result.get('columns')[:5]}...")
            action = Action(tool_name='read_file', tool_input={'file_path': data_path})
            observation = Observation(tool_name='read_file', result=result, success=True)
            step = ReActStep(thought=thought, action=action, observation=observation)
            agent.agent.history.append(step)
        else:
            print(f"Failed to load data: {result.get('error')}")

    print("\nEnter your visualization request:")
    print("-"*60)

    while True:
        try:
            user_input = input("\n> ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if user_input.lower() == 'help':
                print("""
Available commands:
  - Natural language requests for visualizations
  - Examples:
    * "Create a bar plot for gene X, Y, Z"
    * "Show correlation between gene A and gene B"
    * "Generate a heatmap for top 50 genes"
    * "Make a volcano plot for differential expression"
  """)
                continue

            if not user_input:
                continue

            result = agent.run(user_input)

            print("\n" + "="*60)
            print("✅ 可视化结果")
            print("="*60)
            print(format_result(result))
            print("="*60)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == '__main__':
    main()
