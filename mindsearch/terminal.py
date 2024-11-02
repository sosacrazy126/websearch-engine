from datetime import datetime
import sys

from rich.console import Console
from rich.panel import Panel
from rich.traceback import install
from rich.markdown import Markdown

from mindsearch.agent import init_agent
from mindsearch.agent.mindsearch_agent import MindSearchAgent, MindSearchProtocol, WebSearchGraph

# Install rich traceback handler
install(show_locals=True)

# Initialize rich console
console = Console()

def initialize_agent() -> tuple[MindSearchAgent, WebSearchGraph]:
    """Initialize the MindSearch agent and graph with error handling."""
    try:
        agent = init_agent(lang='en', model_format='gpt4', search_engine='DuckDuckGoSearch')
        graph = WebSearchGraph()
        return agent, graph
    except Exception as e:
        console.print(Panel(f"[red]Error initializing agent: {str(e)}[/red]"))
        sys.exit(1)

def chat_loop(agent: MindSearchAgent, graph: WebSearchGraph):
    """Run an interactive chat loop with the agent."""
    console.print(Panel("[blue]Welcome to MindSearch! Type 'exit' to quit.[/blue]"))
    
    while True:
        try:
            # Get user input
            user_input = console.input("[green]You:[/green] ")
            
            # Check for exit command
            if user_input.lower() in ['exit', 'quit']:
                console.print("[yellow]Goodbye![/yellow]")
                break
                
            # Skip empty inputs
            if not user_input.strip():
                continue
                
            # Process the chat and display only final response
            console.print("\n[cyan]Assistant:[/cyan]")
            final_response = None
            
            for agent_return in agent.stream_chat(user_input):
                if isinstance(agent_return, tuple) and len(agent_return) == 3:
                    node_name, node, _ = agent_return
                    if node_name == 'response' and 'content' in node:
                        final_response = node['content']
            
            if final_response:
                console.print(Markdown(final_response))
            console.print("\n")  # Add spacing between interactions
            
        except Exception as e:
            console.print(Panel(f"[red]Error during chat: {str(e)}[/red]"))
            continue

if __name__ == "__main__":
    agent, graph = initialize_agent()
    chat_loop(agent, graph)