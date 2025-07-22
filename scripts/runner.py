#!/usr/bin/env python3
"""
Credit Analyzer CLI

A command-line interface for credit risk analysis and reporting.
Follows the specification in docs/CLI_PROMPT.md
"""

import argparse
import logging
import os
import subprocess
import sys
import time
import os
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the main engine and dependencies
from src.credit_health_engine import CreditHealthEngine
from src.dependency_checker import check_dependencies

# Import CLI enhancement libraries
try:
    from colorama import init, Fore, Style, Back
    from pyfiglet import Figlet
    from tqdm import tqdm
    from rich.console import Console
    from rich.table import Table
    from rich.progress import track
    from rich.panel import Panel
    from rich.text import Text
    HAS_CLI_DEPS = True
except ImportError:
    HAS_CLI_DEPS = False

# Initialize colorama
init(autoreset=True)

# Configure logging
logger = logging.getLogger(__name__)
console = Console() if HAS_CLI_DEPS else None

# Constants
DEFAULT_DATA_DIR = Path('source_data')
DEFAULT_OUTPUT_DIR = Path('output')

# Color definitions
class Colors:
    """Terminal color codes for consistent theming."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class CreditAnalyzerCLI:
    """Main CLI application for Credit Analyzer.
    
    Provides both manual and automated modes for credit risk analysis,
    following the specification in docs/CLI_PROMPT.md
    """
    
    def __init__(self, args=None):
        """Initialize the CLI application."""
        self.args = args or {}
        self.engine = None
        self.running = True
        
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def display_banner(self):
        """Display the creditX ASCII banner."""
        self.clear_screen()
        if HAS_CLI_DEPS:
            f = Figlet(font='slant')
            banner = f.renderText('creditX')
            console.print(f"[bold cyan]{banner}[/]")
            console.print("Credit Analyzer Tool - Interactive CLI\n", style="dim")
        else:
            print(f"{Colors.CYAN}{Colors.BOLD}creditX - Credit Analyzer Tool{Colors.END}\n")
    
    def display_menu(self, title: str, options: List[Tuple[str, str]], 
                    prompt: str = "Enter your choice: ") -> str:
        """Display a menu and get user input.
        
        Args:
            title: Title of the menu
            options: List of (key, description) tuples for menu items
            prompt: Prompt to display for user input
            
        Returns:
            str: The selected menu item key
            
        Raises:
            SystemExit: If running in non-interactive mode (piped input)
        """
        self.clear_screen()
        self.display_banner()
        
        # Check if running in non-interactive mode
        if not sys.stdin.isatty():
            print("\nError: This is an interactive application and cannot be run with piped input.")
            print("Please run the script directly without piping input.")
            print("Example: python runner.py")
            sys.exit(1)
        
        # Create a formatted menu
        menu_lines = []
        menu_lines.append(f"{Colors.BOLD}{title}{Colors.END}\n")
        
        # Add menu options
        for i, (key, desc) in enumerate(options, 1):
            menu_lines.append(f"{Colors.CYAN}[{i}]{Colors.END} {desc}")
        
        # Add quit instruction
        menu_lines.append(f"\n{Colors.YELLOW}[q] Quit{Colors.END}")
        
        # Display the menu
        menu_text = "\n".join(menu_lines)
        
        while True:
            # Clear screen and display banner
            self.clear_screen()
            self.display_banner()
            
            # Print the menu
            print(menu_text)
            
            try:
                # Get user input
                choice = input(f"\n{prompt}").strip().lower()
                
                # Handle quit command
                if choice == 'q':
                    return 'q'
                
                # Validate numeric input
                if choice.isdigit():
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(options):
                        return options[choice_num - 1][0]
                
                # If we get here, the input was invalid
                print(f"{Colors.RED}\nInvalid choice. Please enter a number between 1 and {len(options)} or 'q' to quit.{Colors.END}")
                input("\nPress Enter to continue...")
                
            except (EOFError, KeyboardInterrupt):
                print("\nOperation cancelled by user")
                return 'q'
    
    def show_loading(self, message: str, duration: int = 2):
        """Display a loading animation or message.
        
        Args:
            message: The message to display while loading
            duration: How long to show the loading message (in seconds)
        """
        self.clear_screen()
        self.display_banner()
        
        # Always show the loading message
        loading_text = f"{Colors.CYAN}⏳ {message}...{Colors.END}"
        print(loading_text)
        
        # If we have a duration, show a simple spinner
        if duration > 0:
            spinner = ['|', '/', '-', '\\']
            start_time = time.time()
            i = 0
            
            try:
                while time.time() - start_time < duration:
                    # Move cursor up one line and clear it
                    sys.stdout.write('\033[F\033[K')
                    # Print the loading message with spinner
                    sys.stdout.write(f"{loading_text} {spinner[i % len(spinner)]}")
                    sys.stdout.flush()
                    time.sleep(0.1)
                    i += 1
                # Clear the loading line when done
                sys.stdout.write('\033[F\033[K\033[F')
                sys.stdout.flush()
            except (KeyboardInterrupt, Exception):
                # Make sure we reset the cursor if interrupted
                sys.stdout.write('\n')
                sys.stdout.flush()
        else:
            # Just print a newline if no duration
            print()
    
    def show_success(self, message: str):
        """Display a success message."""
        self.clear_screen()
        self.display_banner()
        print(f"{Colors.GREEN}✓ {message}{Colors.END}\n")
        time.sleep(1)  # Pause briefly so the user can read the message
    
    def show_error(self, message: str):
        """Display an error message."""
        self.clear_screen()
        self.display_banner()
        print(f"{Colors.RED}✗ ERROR: {message}{Colors.END}\n")
        input("Press Enter to continue...")
    
    def show_warning(self, message: str):
        """Display a warning message."""
        print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")
        logger.warning(message)
        
    def show_info(self, message: str):
        """Display an informational message.
        
        Args:
            message: The message to display
        """
        print(f"{Colors.CYAN}ℹ {message}{Colors.END}")
        logger.info(message)
    
    def confirm_action(self, message: str) -> bool:
        """Ask for confirmation before proceeding."""
        if HAS_CLI_DEPS:
            return console.input(f"[yellow]{message} (y/N): [/]").strip().lower() == 'y'
        return input(f"{Colors.YELLOW}{message} (y/N): {Colors.END}").strip().lower() == 'y'
    
    def main_menu(self):
        """Display the main menu and handle user input.
        
        Follows the specification in docs/CLI_PROMPT.md with three main options:
        1. Manual Mode - Full control with step-by-step execution
        2. Auto Mode - Simplified flow with minimal input
        3. Clear - Remove generated output files
        """
        while self.running:
            options = [
                ('1', 'Manual Mode - Full control with step-by-step execution'),
                ('2', 'Auto Mode - Simplified flow with minimal input'),
                ('3', 'Clear - Remove generated output files'),
                ('q', 'Quit')
            ]
            
            choice = self.display_menu(
                "CREDIT ANALYZER - MAIN MENU",
                options,
                "\nSelect an option (1-3) or 'q' to quit: "
            )
            
            if choice == '1':
                self.manual_mode()
            elif choice == '2':
                self.auto_mode()
            elif choice == '3':
                self.clear_output()
            elif choice == 'q':
                self.running = False
    
    def manual_mode(self):
        """Run in manual mode with step-by-step execution.
        
        Follows the specification in docs/CLI_PROMPT.md with these steps:
        1. Load Data - Load raw data from predefined sources
        2. Clean and Analyze - Perform data cleaning and analysis
        3. Unique Agent Lookup - Find and display agent information
        4. Regional Top & Bottom Agent Analysis - Generate performance reports
        """
        self.clear_screen()
        self.show_loading("Entering Manual Mode")
        
        while True:
            options = [
                ('1', 'Load Data - Load raw data from predefined sources'),
                ('2', 'Clean and Analyze - Perform data cleaning and analysis'),
                ('3', 'Agent Lookup - Find and display agent information'),
                ('4', 'Regional Analysis - Top/Bottom agents by region'),
                ('q', 'Back to Main Menu')
            ]
            
            choice = self.display_menu(
                "MANUAL MODE - Step by Step Execution",
                options,
                "\nSelect an option (1-4) or 'q' to return to main menu: "
            )
            
            if choice == '1':
                self.load_data()
            elif choice == '2':
                self.clean_and_analyze()
            elif choice == '3':
                self.agent_lookup()
            elif choice == '4':
                self.regional_analysis()
            elif choice == 'q':
                break
    
    def auto_mode(self):
        """Run in auto mode with minimal user interaction.
        
        Follows the specification in docs/CLI_PROMPT.md with these steps:
        1. Perform unique agent lookup with default parameters
        2. Generate Top 10 and Bottom 10 agents by region
        3. Save results to output files
        """
        self.clear_screen()
        self.show_loading("Entering Auto Mode - Simplified Workflow")
        
        try:
            # Step 1: Perform agent lookup with default parameters
            self.show_loading("Performing agent lookup...")
            if not self.agent_lookup(auto_mode=True):
                self.show_error("Agent lookup failed. Check logs for details.")
                return False
            
            # Step 2: Generate reports (includes regional and top/bottom agents)
            self.show_loading("Generating reports...")
            
            # First, ensure we have the engine initialized
            if not hasattr(self, 'engine') or not self.engine:
                self.show_error("Engine not initialized. Please load data first.")
                return False
                
            # Generate all reports
            try:
                success = self.engine.generate_reports()
                if success:
                    self.show_success("Reports generated successfully!")
                else:
                    self.show_error("Failed to generate some reports. Check logs for details.")
                    return False
            except Exception as e:
                self.show_error(f"Error generating reports: {str(e)}")
                if self.args.get('debug'):
                    import traceback
                    traceback.print_exc()
                return False
            
            # Step 3: Show completion message with output locations
            self.show_success("\nAuto mode completed successfully!")
            print("\nOutput files saved to:")
            print(f"- Agent Lookup: {DEFAULT_OUTPUT_DIR}/agent_lookup/")
            print(f"- Reports: {DEFAULT_OUTPUT_DIR}/reports/")
            
            return True
            
        except Exception as e:
            self.show_error(f"An error occurred in auto mode: {str(e)}")
            logger.exception("Error in auto mode")
            return False
    
    def clean_and_analyze(self) -> bool:
        """
        Run the complete analysis pipeline including feature engineering,
        agent classification, and report generation.
        
        Returns:
            bool: True if analysis was successful, False otherwise
        """
        # Check if data is loaded
        if not hasattr(self, 'engine') or self.engine is None:
            self.show_error("Please load data first!")
            return False
            
        try:
            # Step 1: Engineer features
            self.show_loading("Engineering features...")
            if not self.engine.engineer_features():
                self.show_error("Feature engineering failed. Check logs for details.")
                return False
                
            # Step 2: Classify agents
            self.show_loading("Classifying agents...")
            if not self.engine.classify_agents():
                self.show_error("Agent classification failed. Check logs for details.")
                return False
                
            # Step 3: Generate reports
            self.show_loading("Generating reports...")
            if hasattr(self.engine, 'generate_reports') and callable(self.engine.generate_reports):
                # Handle both return types: bool and tuple[bool, str]
                result = self.engine.generate_reports()
                if isinstance(result, tuple):
                    report_success, report_message = result
                else:
                    report_success = result
                    report_message = ""
                
                if report_success:
                    self.show_success("Analysis completed successfully!")
                    if report_message:
                        self.show_info(f"Report: {report_message}")
                    return True
                else:
                    self.show_error(f"Report generation failed: {report_message}")
                    return False
            else:
                self.show_warning("Report generation not available. Analysis completed without reports.")
                return True
                
        except Exception as e:
            self.show_error(f"An error occurred during analysis: {str(e)}")
            logger.exception("Error in clean_and_analyze")
            return False
            
    def load_data(self) -> bool:
        """Load and validate data from source files."""
        try:
            self.show_loading("Checking data directory...")
                
            # Check if data directory exists
            if not DEFAULT_DATA_DIR.exists():
                self.show_error(f"Data directory not found: {DEFAULT_DATA_DIR}")
                self.show_warning("Please ensure your data files are in the 'source_data' directory.")
                return False
                    
            # Import REQUIRED_FILES from credit_health_engine
            from src.credit_health_engine import REQUIRED_FILES
                
            # Check for missing files
            missing_files = [f for f in REQUIRED_FILES.keys() 
                          if not (DEFAULT_DATA_DIR / f).exists()]
                
            if missing_files:
                self.show_error(f"Missing required data files: {', '.join(missing_files)}")
                self.show_warning(f"Please ensure these files are in the '{DEFAULT_DATA_DIR}' directory.")
                return False
                    
            # Initialize the engine if not already done
            if self.engine is None:
                self.engine = CreditHealthEngine()
                    
            # Load and validate data
            self.show_loading("Loading and validating data...")
            try:
                self.engine.load_data()
                self.show_success("Data loaded and validated successfully!")
                return True
            except Exception as e:
                self.show_error(f"Error loading data: {str(e)}")
                logger.exception("Data loading failed")
                return False
                    
        except Exception as e:
            self.show_error(f"An unexpected error occurred: {str(e)}")
            logger.exception("Error in load_data")
            return False
    
    def run_analysis(self) -> bool:
        """Run the analysis pipeline."""
        if not hasattr(self, 'engine') or self.engine is None:
            self.show_error("Please load data first!")
            return False
                
        try:
            self.show_loading("Running analysis...")
                
            # Run feature engineering
            self.engine.calculate_features()
                
            # Classify agents
            self.engine.classify_agents()
                
            self.show_success("Analysis completed successfully!")
            return True
                
        except Exception as e:
            self.show_error(f"Error during analysis: {str(e)}")
            logger.exception("Analysis failed")
            return False
        
    def agent_lookup(self, auto_mode: bool = False):
        """Look up agent information using the lookup_agent_profiles script.
        
        Args:
            auto_mode: If True, run in automated mode with default parameters.
            1. Perform unique agent lookup with default parameters
            2. Generate Top 10 and Bottom 10 agents by region
            3. Save results to output files
                      If False, prompt user for input.
        """
        try:
            if auto_mode:
                # In auto mode, use a default agent ID or implement batch processing
                self.show_loading("Running agent lookup in auto mode...")
                # Default implementation - can be enhanced based on requirements
                self.show_warning("Auto mode for agent lookup is not fully implemented.")
                self.show_info("Defaulting to interactive agent lookup...")
                # Fall back to interactive mode
                auto_mode = False
            
            if not auto_mode:
                print("\n" + "="*50)
                print("AGENT LOOKUP")
                print("="*50)
                
                # Get agent ID from user
                agent_id = input("\nEnter Agent ID (or 'q' to cancel): ").strip()
                
                if agent_id.lower() == 'q':
                    self.show_warning("Operation cancelled by user.")
                    return False
                    
                if not agent_id:
                    self.show_error("No agent ID provided.")
                    return False
                    
                # Import the lookup function directly instead of using subprocess
                try:
                    import pandas as pd
                    from scripts.lookup_agent_profiles import load_agents, load_sales, load_repayment_metrics, AgentReporter, AgentClassifier
                    
                    # Load the data
                    agents = load_agents()
                    sales_data = load_sales()
                    repay_metrics = load_repayment_metrics()
                    
                    # Standardize column names
                    agents.rename(columns={'Bzid': 'BZID'}, inplace=True)
                    agents['BZID'] = agents['BZID'].astype(str).str.strip()
                    
                    # Merge with sales data if available
                    if not sales_data.empty and 'Bzid' in sales_data.columns:
                        sales_data.rename(columns={'Bzid': 'BZID'}, inplace=True)
                        agents = pd.merge(agents, sales_data, on='BZID', how='left')
                    
                    # Merge with repayment metrics if available
                    if not repay_metrics.empty and 'Bzid' in repay_metrics.columns:
                        repay_metrics.rename(columns={'Bzid': 'BZID'}, inplace=True)
                        agent_data = pd.merge(agents, repay_metrics, on='BZID', how='left')
                    else:
                        agent_data = agents
                    
                    # Find the agent
                    agent = agent_data[agent_data['BZID'].astype(str).str.strip() == agent_id]
                    
                    if agent.empty:
                        self.show_error(f"Agent with ID {agent_id} not found.")
                        return False
                    
                    # Get the agent data as a Series
                    agent_series = agent.iloc[0]
                    
                    # Initialize reporter with the series data
                    reporter = AgentReporter(agent_series.to_dict())
                    analysis = reporter.analyze()
                    
                    # Classify the agent
                    try:
                        classifier = AgentClassifier()
                        # Pass the Series directly to maintain index/name
                        classification = classifier.classify_agent(agent_series)
                        analysis['classification'] = classification
                        
                        # Print the profile
                        from scripts.lookup_agent_profiles import print_agent_profile
                        print("\n" + "="*50)
                        print(f"AGENT PROFILE - ID: {agent_id}")
                        print("="*50)
                        # Convert series to dict for the print_agent_profile function
                        agent_data = agent_series.to_dict()
                        print_agent_profile(agent_id, agent_data, analysis)
                        print("\nPress Enter to continue...")
                        input()  # Wait for user to press Enter
                        return True
                        
                    except Exception as e:
                        self.show_error(f"Error analyzing agent: {str(e)}")
                        logger.exception("Error in agent analysis")
                        return False
                    
                except subprocess.TimeoutExpired:
                    self.show_error("Agent lookup timed out after 30 seconds.")
                    return False
                except Exception as e:
                    self.show_error(f"Error during agent lookup: {str(e)}")
                    logger.exception("Agent lookup failed")
                    return False
            else:
                # Placeholder for auto mode implementation
                self.show_info("Auto mode agent lookup would run here.")
                return True
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            return False
        except Exception as e:
            self.show_error(f"An unexpected error occurred: {str(e)}")
            logger.exception("Error in agent_lookup")
            return False
            success = False
                
        return success

    def _generate_regional_report(self) -> bool:
        """Generate a regional performance report."""
        if not hasattr(self, 'engine') or not self.engine:
            return False
                    
        # Get available regions if the method exists
        try:
            if hasattr(self.engine, 'get_available_regions'):
                regions = self.engine.get_available_regions()
                if regions:
                    # Show region selection
                    region_options = [(r, r) for r in regions] + [("all", "All Regions")]
                    choice = self.display_menu("Select Region", region_options)
                    
                    if choice == 'q':
                        return False
                    
                    self.show_loading(f"Generating report for {choice}")
                else:
                    self.show_loading("Generating report for all regions")
            else:
                self.show_loading("Generating reports")
                
            # Use the main generate_reports method
            success = self.engine.generate_reports()
            
            if success:
                self.show_success("Reports generated successfully!")
                return True
            else:
                self.show_error("Failed to generate reports")
                return False
                
        except Exception as e:
            self.show_error(f"Error generating reports: {str(e)}")
            if self.args.get('debug'):
                import traceback
                traceback.print_exc()
            return False

    def _generate_top_bottom_report(self):
        """Generate top/bottom agents report.
        
        Note: This is now handled by the main generate_reports method.
        """
        self.show_warning("This method is deprecated. Use generate_reports() instead.")
        return False
        
    def regional_analysis(self):
        """Perform regional analysis and generate reports with top/bottom agents.
        
        This method will:
        1. Check if analysis has been performed
        2. Let user select a region or view all regions
        3. Show top 10 and bottom 10 agents for the selected region
        4. Display tier distribution and statistics
        """
        if not hasattr(self, 'engine') or not hasattr(self.engine, 'classified_agents'):
            self.show_error("Please run analysis first (option 2 in manual mode)")
            return False
            
        try:
            # Generate reports first
            self.show_loading("Generating reports...")
            report_path = self.engine.generate_reports()
            if not report_path:
                self.show_error("Failed to generate analysis reports")
                return False
                
            df = self.engine.classified_agents
            
            # Load region data from Region_contact.xlsx
            has_region = False
            try:
                region_file = Path('source_data/Region_contact.xlsx')
                if region_file.exists():
                    region_df = pd.read_excel(region_file)
                    if 'Region' in region_df.columns:
                        # Get unique regions, remove duplicates and sort
                        unique_regions = region_df['Region'].dropna().astype(str).unique().tolist()
                        unique_regions = sorted(list(set(unique_regions)))
                        
                        if unique_regions:
                            has_region = True
                            # Add 'All Regions' as the first option
                            regions = ['All Regions'] + unique_regions
                            
                            # Let user select a region
                            self.clear_screen()
                            self.display_banner()
                            print(f"{Colors.CYAN}🌍 Select a region to analyze:{Colors.END}\n")
                            
                            for i, region in enumerate(regions, 1):
                                print(f"{i}. {region}")
                            
                            print("\n0. Back to main menu")
                        else:
                            self.show_warning("No valid regions found in Region_contact.xlsx. Using 'All Regions'.")
                            regions = ['All Regions']
                    else:
                        self.show_warning("No 'Region' column found in Region_contact.xlsx. Using 'All Regions'.")
                        regions = ['All Regions']
                else:
                    self.show_warning("Region_contact.xlsx not found. Using 'All Regions'.")
                    regions = ['All Regions']
            except Exception as e:
                self.show_warning(f"Error loading region data: {str(e)}. Using 'All Regions'.")
                logger.exception("Error in regional analysis")
                regions = ['All Regions']
                
            if has_region:
                while True:
                    try:
                        choice = input("\nEnter your choice (number): ").strip()
                        if choice == '0':
                            return True
                            
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < len(regions):
                            selected_region = regions[choice_idx]
                            break
                            
                        self.show_warning("Invalid choice. Please try again.")
                        
                    except (ValueError, IndexError):
                        self.show_warning("Please enter a valid number.")
            else:
                selected_region = 'All Regions'
            
            # Filter data by selected region if not 'All Regions'
            if has_region and selected_region != 'All Regions':
                region_data = df[df['Region'] == selected_region].copy()
                region_name = selected_region
            else:
                region_data = df.copy()
                region_name = 'All Regions'
            
            # Clear screen and show analysis
            self.clear_screen()
            self.display_banner()
            print(f"{Colors.GREEN}✓ Regional Analysis: {region_name}{Colors.END}\n")
            
            # Show tier distribution for the region
            if not region_data.empty:
                # Tier distribution
                tier_dist = region_data['tier'].value_counts().sort_index()
                total_agents = len(region_data)
                
                print(f"{Colors.CYAN}📊 Tier Distribution:{Colors.END}")
                print("-" * 50)
                for tier, count in tier_dist.items():
                    percentage = (count / total_agents) * 100
                    print(f"{tier}: {count} agents ({percentage:.1f}%)")
                print()
                
                # Top 10 agents by credit health score
                if 'credit_health_score' in region_data.columns:
                    # Top 10 agents
                    top_agents = region_data.nlargest(10, 'credit_health_score')
                    print(f"{Colors.GREEN}🏆 Top 10 Agents by Credit Health Score:{Colors.END}")
                    print("-" * 70)
                    print("ID          Score  Tier  Region")
                    print("-" * 70)
                    for _, row in top_agents.iterrows():
                        print(f"{row['Bzid']:<10} {row['credit_health_score']:5.1f}  {row['tier']:^4}  {row.get('Region', 'N/A')}")
                    
                    # Bottom 10 agents
                    bottom_agents = region_data.nsmallest(10, 'credit_health_score')
                    print(f"\n{Colors.RED}⚠  Bottom 10 Agents by Credit Health Score:{Colors.END}")
                    print("-" * 70)
                    print("ID          Score  Tier  Region")
                    print("-" * 70)
                    for _, row in bottom_agents.iterrows():
                        print(f"{row['Bzid']:<10} {row['credit_health_score']:5.1f}  {row['tier']:^4}  {row.get('Region', 'N/A')}")
                    
                    print()
                
                # Additional statistics
                if 'credit_utilization' in region_data.columns:
                    avg_utilization = region_data['credit_utilization'].mean()
                    print(f"{Colors.CYAN}📊 Average Credit Utilization:{Colors.END} {avg_utilization:.1f}%")
                
                if 'repayment_score' in region_data.columns:
                    avg_repayment = region_data['repayment_score'].mean()
                    print(f"{Colors.CYAN}📈 Average Repayment Score:{Colors.END} {avg_repayment:.1f}/100")
                
                print(f"\n{Colors.GREEN}✓ Analysis complete for {region_name}{Colors.END}")
            else:
                self.show_warning(f"No data available for {region_name}")
            
            print("\nPress Enter to return to the main menu...")
            input()
            return True
                
        except Exception as e:
            self.show_error(f"Error during regional analysis: {str(e)}")
            logger.exception("Error in regional_analysis")
            return False
            
            self.show_success("Successfully cleared output files!")
            return True
            
        except Exception as e:
            self.show_error(f"Failed to clear output directory: {str(e)}")
            logger.exception("Error clearing output directory")
            return False
        input("\nPress Enter to continue...")

    def run(self):
        """Run the CLI application."""
        self.display_banner()
        
        if not HAS_CLI_DEPS:
            self.show_warning(
                "Some CLI enhancements are not available. "
                "Install with: pip install colorama pyfiglet tqdm rich"
            )
            time.sleep(2)
            
        try:
            self.main_menu()
        except KeyboardInterrupt:
            self.show_warning("\nOperation cancelled by user")
        except Exception as e:
            self.show_error(f"An error occurred: {str(e)}")
            if self.args.get('debug'):
                import traceback
                traceback.print_exc()
        finally:
            self.show_success("Thank you for using creditX. Goodbye!")
            if HAS_CLI_DEPS:
                console.print("\n[dim]Press any key to exit...[/]")
                try:
                    import msvcrt
                    msvcrt.getch()
                except ImportError:
                    input()
            else:
                input("\nPress Enter to exit...")

def parse_arguments():
    """Parse command line arguments for the creditX CLI.
    
    Returns:
        dict: Parsed command line arguments as a dictionary.
    """
    parser = argparse.ArgumentParser(description='CreditX - Credit Analyzer Tool')
    
    # Add arguments
    parser.add_argument('--data-dir', type=str, default=str(DEFAULT_DATA_DIR),
                      help='Directory containing input data files')
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR),
                      help='Directory to store output files')
    parser.add_argument('--force', action='store_true',
                      help='Force overwrite of existing files')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug logging')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert to dictionary
    return {
        'data_dir': Path(args.data_dir),
        'output_dir': Path(args.output_dir),
        'force': args.force,
        'debug': args.debug
    }

# Constants
DEFAULT_DATA_DIR = Path('source_data')
DEFAULT_OUTPUT_DIR = Path('output')

# Ensure UTF-8 output on Windows
if sys.platform.startswith('win'):
    import io
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, 
            encoding='utf-8', 
            errors='replace'
        )
    if sys.stderr.encoding != 'utf-8':
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, 
            encoding='utf-8', 
            errors='replace'
        )

# Set up logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)  # Default log level

# Clear any existing handlers
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
console_handler.setLevel(logging.INFO)

# Create file handler
log_file = Path('creditx.log')
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
file_handler.setLevel(logging.DEBUG)  # Always log debug to file

# Add handlers to root logger
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

# Log initialization
logger = logging.getLogger(__name__)
logger.info("Logging initialized")
logger.debug(f"Log file: {log_file.absolute()}")

def setup_logging(debug=False):
    """Set up logging for the creditX CLI."""
    logger.info("Setting up logging...")
    
    # Set log level based on debug flag
    if debug:
        root_logger.setLevel(logging.DEBUG)
        for handler in root_logger.handlers:
            handler.setLevel(logging.DEBUG)
    else:
        root_logger.setLevel(logging.INFO)
        for handler in root_logger.handlers:
            handler.setLevel(logging.INFO)

def main():
    """Main entry point for the creditX CLI."""
    # Parse command line arguments
    args = parse_arguments()
    debug_mode = args.get('debug', False)
    
    # Set up logging
    setup_logging(debug=debug_mode)
    logger = logging.getLogger(__name__)
    logger.info("Starting creditX CLI application")
    logger.debug(f"Command line arguments: {args}")
    
    try:
        # Initialize and run the CLI
        logger.info("Initializing CreditAnalyzerCLI")
        cli = CreditAnalyzerCLI(args)
        logger.info("Running CLI application")
        return cli.run()
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 0
    except Exception as e:
        logger.critical(f"Error: {str(e)}", exc_info=debug_mode)
        return 1

def clear_output(output_dir: Path, force: bool = False) -> int:
    """Clear all output files from the output directory.
    
    Args:
        output_dir: Path to the output directory
        force: If True, skip confirmation prompt
        
    Returns:
        int: Status code (0 for success, non-zero for error)
    """
    try:
        if not output_dir.exists():
            print(f"Output directory does not exist: {output_dir}")
            return 0
            
        if not output_dir.is_dir():
            print(f"Path exists but is not a directory: {output_dir}")
            return 1
            
        if not force:
            confirm = input(f"Are you sure you want to clear all files in {output_dir}? (y/N): ")
            if confirm.lower() != 'y':
                print("Operation cancelled.")
                return 0
                
        # Remove all files in the output directory
        for item in output_dir.glob('*'):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                import shutil
                shutil.rmtree(item)
                
        print(f"Successfully cleared {output_dir}")
        return 0
        
    except Exception as e:
        print(f"Error clearing output directory: {e}")
        return 1

def setup_environment(args):
    """Set up the execution environment.
    
    Args:
        args: Dictionary containing command line arguments
        
    Returns:
        bool: True if setup was successful, False otherwise
    """
    logger.info("Setting up environment...")
    
    # Ensure data directory exists
    try:
        data_dir = args['data_dir']
        output_dir = args['output_dir']
        
        # Create directories if they don't exist
        data_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        setup_logging(debug=args.get('debug', False))
        
        # Verify data directory contains required files
        required_files = [
            'credit_Agents.xlsx',
            'Credit_history_sales_vs_credit_sales.xlsx',
            'DPD.xlsx',
            'Region_contact.xlsx',
            'sales_data.xlsx'
        ]
        
        missing_files = [f for f in required_files if not (data_dir / f).exists()]
        if missing_files:
            logger.warning(f"The following required files are missing: {', '.join(missing_files)}")
            logger.warning(f"Please ensure all required files are in the data directory: {data_dir}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error setting up environment: {e}", exc_info=True)
        return False

if __name__ == '__main__':
    sys.exit(main())
