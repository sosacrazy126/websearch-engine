from datetime import datetime
import os
import logging
import sys
from typing import Optional
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.logging import RichHandler
from rich.traceback import install
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from lagent.actions import ActionExecutor, BingBrowser
from lagent.llms import GPTAPI
from mindsearch.agent.mindsearch_agent import (MindSearchAgent, MindSearchProtocol)
from mindsearch.agent.mindsearch_prompt import (
    FINAL_RESPONSE_EN, GRAPH_PROMPT_EN,
    searcher_context_template_en, searcher_input_template_en,
    searcher_system_prompt_en)
from mindsearch.agent.mindsearch_agent import WebSearchGraph

# Install rich traceback handler
install(show_locals=True)

# Initialize rich console
console = Console()

# Initialize logging with rich handler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, tracebacks_show_locals=True)]
)
logger = logging.getLogger("mindsearch")

def create_layout() -> Layout:
    """Create the layout for the terminal interface."""
    layout = Layout(name="root")
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3)
    )
    return layout

def initialize_agent() -> tuple[MindSearchAgent, WebSearchGraph]:
    """Initialize the MindSearch agent and graph with error handling."""
    try:
        # Configure GPT settings
        llm = GPTAPI(
            model_type=os.environ.get('OPENAI_MODEL', 'gpt-4-mini'),
            key=os.environ.get('OPENAI_API_KEY', 'YOUR OPENAI API KEY'),
            openai_api_base=os.environ.get('OPENAI_API_BASE', 'https://api.openai.com/v1/chat/completions')
        )

        agent = MindSearchAgent(
            llm=llm,
            protocol=MindSearchProtocol(
                meta_prompt=datetime.now().strftime('The current date is %Y-%m-%d.'),
                interpreter_prompt=GRAPH_PROMPT_EN,
                response_prompt=FINAL_RESPONSE_EN),
            searcher_cfg=dict(
                llm=llm,
                plugin_executor=ActionExecutor(
                    BingBrowser(searcher_type='DuckDuckGoSearch', topk=6)),
                protocol=MindSearchProtocol(
                    meta_prompt=datetime.now().strftime('The current date is %Y-%m-%d.'),
                    plugin_prompt=searcher_system_prompt_en,
                ),
                template=dict(
                    input=searcher_input_template_en,
                    context=searcher_context_template_en
                )
            ),
            max_turn=10
        )
        
        graph = WebSearchGraph()
        return agent, graph
    
    except Exception as e:
        console.print(Panel(f"[red]Error initializing agent: {str(e)}[/red]"))
        sys.exit(1)

def clean_response(response: str) -> str:
    """Clean the response text by removing duplicates and formatting."""
    # Split into words and remove duplicates while maintaining order
    words = response.split()
    cleaned_words = []
    seen = set()
    
    for word in words:
        if word not in seen:
            cleaned_words.append(word)
            seen.add(word)
    
    # Rejoin words and format properly
    cleaned_text = ' '.join(cleaned_words)
    return cleaned_text

def format_response(response: str, node_info: dict) -> Panel:
    """Format the agent's response with additional information."""
    # Simple text cleanup - remove duplicate concatenated text
    response = ' '.join(response.split())  # This splits on whitespace and rejoins, removing extra spaces
    
    table = Table(show_header=True, header_style="bold magenta", padding=(0, 1))
    table.add_column("Response", style="green", no_wrap=False)
    
    # Add the cleaned response
    table.add_row(response)
    
    return Panel(
        table,
        title="[bold blue]Agent Response[/bold blue]",
        border_style="blue"
    )

def handle_agent_response(agent_return) -> str:
    """Handle agent response with error checking and duplicate prevention."""
    try:
        if isinstance(agent_return, tuple):
            agent_return = agent_return[0]  # Extract the first element if it's a tuple
            
        if hasattr(agent_return, 'response'):
            return agent_return.response
        else:
            logger.warning(f"Unexpected agent_return format: {agent_return}")
            return ""
            
    except Exception as e:
        logger.error(f"Error processing agent response: {str(e)}")
        return ""

def save_chat_history(responses):
    """Save chat history to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_history_{timestamp}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(responses, f, indent=2)
        return f"Chat history saved to {filename}"
    except Exception as e:
        return f"Error saving chat history: {e}"

def main():
    """Main function to run the enhanced terminal interface."""
    console.print(Panel.fit(
        "[bold blue]MindSearch Terminal Interface[/bold blue]\n"
        "[italic]Type 'exit' or 'quit' to end the session[/italic]",
        border_style="bold blue"
    ))

    try:
        agent, graph = initialize_agent()
        chat_history = []  # Store chat history
        
        while True:
            user_input = Prompt.ask("\n[bold green]You")
            if user_input.lower() in ['exit', 'quit']:
                # Ask if user wants to save chat history
                save_option = Prompt.ask("\nSave chat history? [y/N]")
                if save_option.lower() == 'y':
                    result = save_chat_history(chat_history)
                    console.print(f"[yellow]{result}[/yellow]")
                console.print("[yellow]Goodbye![/yellow]")
                break
                
            response = ""
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Processing...", total=None)
                
                try:
                    for agent_return in agent.stream_chat(user_input):
                        if isinstance(agent_return, tuple):
                            agent_return = agent_return[0]
                        
                        if hasattr(agent_return, 'response'):
                            response = agent_return.response
                            
                    # Add to chat history
                    chat_history.append({
                        "user": user_input,
                        "agent": response,
                        "timestamp": datetime.now().isoformat()
                    })
                            
                    # Add response node and get info
                    graph.add_response_node(node_name='response')
                    response_info = graph.node('response')
                    
                    # Log node info
                    logger.info(f"Response Node Info: {response_info}")
                    
                    # Display formatted response
                    console.print(format_response(response, response_info))
                    
                except Exception as e:
                    console.print(Panel(
                        f"[red]Error processing request: {str(e)}[/red]",
                        title="Error",
                        border_style="red"
                    ))
                    logger.exception("Error in processing request")
                    
    except KeyboardInterrupt:
        # Also offer to save on keyboard interrupt
        save_option = Prompt.ask("\nSave chat history? [y/N]")
        if save_option.lower() == 'y':
            result = save_chat_history(chat_history)
            console.print(f"[yellow]{result}[/yellow]")
        console.print("\n[yellow]Session terminated by user[/yellow]")
        
    except Exception as e:
        console.print(Panel(f"[red]Fatal error: {str(e)}[/red]"))
        logger.exception("Fatal error occurred")
        sys.exit(1)

if __name__ == "__main__":
    main() 
