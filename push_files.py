#!/usr/bin/env python3
"""
Automated Git File Pusher
Automates the process of pushing files one by one to GitHub with error handling and retry logic.
"""

import os
import subprocess
import time
import json
from pathlib import Path
from typing import List, Tuple, Optional

class GitAutomator:
    def __init__(self, max_retries: int = 3, retry_delay: int = 5):
        """
        Initialize GitAutomator with retry settings.
        
        Args:
            max_retries: Maximum number of retry attempts for failed pushes
            retry_delay: Delay between retry attempts in seconds
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.large_file_threshold = 1000  # Lines threshold for "large" files
        self.git_configured = False
        
    def run_command(self, command: List[str], check: bool = True) -> Tuple[bool, str]:
        """
        Run a shell command and return success status and output.
        
        Args:
            command: List of command arguments
            check: Whether to raise exception on failure
            
        Returns:
            Tuple of (success: bool, output: str)
        """
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=check
            )
            return True, result.stdout
        except subprocess.CalledProcessError as e:
            return False, f"Command failed: {e.stderr}"
    
    def check_git_config(self) -> bool:
        """
        Check if git is properly configured with credentials.
        
        Returns:
            bool: True if git is properly configured
        """
        # Check if remote URL contains credentials
        success, remote_url = self.run_command(['git', 'remote', 'get-url', 'origin'])
        if success:
            if '@' in remote_url and 'github.com' in remote_url:
                print("✓ Git remote configured with credentials")
                return True
            else:
                print("✗ Git remote missing credentials")
                print(f"  Current URL: {remote_url}")
                return False
        return False
    
    def get_file_line_count(self, filepath: Path) -> int:
        """
        Get the line count of a file.
        
        Args:
            filepath: Path to the file
            
        Returns:
            Number of lines in the file
        """
        try:
            return sum(1 for _ in open(filepath, encoding='utf-8'))
        except Exception as e:
            print(f"  Warning: Could not count lines in {filepath}: {e}")
            return 0
    
    def commit_and_push_file(self, filepath: Path, message: str) -> bool:
        """
        Commit and push a single file with retry logic.
        
        Args:
            filepath: Path to the file to push
            message: Commit message
            
        Returns:
            bool: True if push was successful
        """
        file_size = self.get_file_line_count(filepath)
        is_large = file_size > self.large_file_threshold
        
        print(f"\n{'='*60}")
        print(f"Processing: {filepath}")
        print(f"Size: {file_size} lines {'(Large file) 📊' if is_large else '(Small file) 📝'}")
        print(f"Message: {message}")
        
        if is_large:
            print(f"⚠️  Large file detected - may require retries")
        
        # Check if file exists
        if not filepath.exists():
            print(f"✗ File not found: {filepath}")
            return False
        
        for attempt in range(1, self.max_retries + 1):
            print(f"\nAttempt {attempt}/{self.max_retries}")
            
            try:
                # Add file
                success, _ = self.run_command(['git', 'add', str(filepath)])
                if not success:
                    print(f"  ✗ Failed to add file")
                    return False
                
                # Commit
                commit_message = f"{message} | Size: {file_size} lines"
                success, output = self.run_command(['git', 'commit', '-m', commit_message])
                if 'nothing to commit' in output.lower():
                    print(f"  ✓ File already committed")
                    return True
                elif not success:
                    print(f"  ✗ Failed to commit: {output}")
                    return False
                
                # Push
                success, output = self.run_command(['git', 'push'])
                if success:
                    print(f"  ✓ Successfully pushed: {filepath}")
                    return True
                else:
                    error_msg = output.lower()
                    if '403' in error_msg or 'forbidden' in error_msg:
                        print(f"  ✗ Authentication error - check credentials")
                    elif 'rpc failed' in error_msg or 'network' in error_msg:
                        print(f"  ✗ Network error - retrying...")
                        if attempt < self.max_retries:
                            time.sleep(self.retry_delay)
                            continue
                    else:
                        print(f"  ✗ Push failed: {output}")
                    
                    # Reset and try again
                    self.run_command(['git', 'reset', '--hard', 'HEAD~1'], check=False)
                    
            except Exception as e:
                print(f"  ✗ Error: {e}")
                self.run_command(['git', 'reset', '--hard', 'HEAD~1'], check=False)
            
            if attempt < self.max_retries:
                print(f"  Waiting {self.retry_delay} seconds before retry...")
                time.sleep(self.retry_delay)
        
        print(f"  ✗ Failed to push after {self.max_retries} attempts")
        return False
    
    def push_files_from_pattern(self, pattern: str, prefix_message: str) -> Tuple[int, int]:
        """
        Push files matching a pattern.
        
        Args:
            pattern: Glob pattern to match files
            prefix_message: Prefix for commit messages
            
        Returns:
            Tuple of (success_count, failure_count)
        """
        files = list(Path('.').glob(pattern))
        print(f"\nFound {len(files)} files matching '{pattern}'")
        
        success_count = 0
        failure_count = 0
        
        for filepath in sorted(files):
            # Skip directories
            if filepath.is_dir():
                continue
                
            # Generate commit message
            rel_path = str(filepath)
            message = f"{prefix_message}: {rel_path}"
            
            # Attempt to push
            if self.commit_and_push_file(filepath, message):
                success_count += 1
            else:
                failure_count += 1
                
        return success_count, failure_count
    
    def interactive_file_selection(self) -> List[Tuple[Path, str]]:
        """
        Interactively select files to push from untracked files.
        
        Returns:
            List of (filepath, message) tuples
        """
        # Get untracked files
        success, output = self.run_command(['git', 'status', '--short'])
        if not success:
            print("Failed to get git status")
            return []
        
        # Parse untracked files (those starting with ??)
        untracked = []
        for line in output.split('\n'):
            if line.startswith('??'):
                filepath = Path(line.strip()[3:]).strip()
                if filepath.exists() and not filepath.is_dir():
                    untracked.append(filepath)
        
        if not untracked:
            print("No untracked files found")
            return []
        
        print(f"\nFound {len(untracked)} untracked files:")
        for i, filepath in enumerate(untracked, 1):
            size = self.get_file_line_count(filepath)
            size_indicator = "📊 Large" if size > self.large_file_threshold else "📝 Small"
            print(f"  {i}. {filepath} ({size}) {size_indicator}")
        
        print("\nEnter file numbers to push (comma-separated, 'all' for all, 'q' to quit):")
        user_input = input("> ").strip()
        
        if user_input.lower() == 'q':
            return []
        elif user_input.lower() == 'all':
            return [(filepath, f"Add {str(filepath)}") for filepath in untracked]
        else:
            try:
                indices = [int(x.strip()) for x in user_input.split(',')]
                selected = []
                for i in indices:
                    if 1 <= i <= len(untracked):
                        filepath = untracked[i-1]
                        message = f"Add {str(filepath)}"
                        selected.append((filepath, message))
                    else:
                        print(f"Skipping invalid index: {i}")
                return selected
            except ValueError:
                print("Invalid input")
                return []
    
    def run(self):
        """
        Main execution method.
        """
        print("="*60)
        print("Git Automated File Pusher")
        print("="*60)
        
        # Check git configuration
        print("\n🔍 Checking git configuration...")
        if not self.check_git_config():
            print("\n⚠️  Git is not properly configured!")
            print("Please update remote URL with your token:")
            print("git remote set-url origin https://TOKEN@github.com/USER/REPO")
            return
        
        # Interactive file selection
        files_to_push = self.interactive_file_selection()
        
        if not files_to_push:
            print("\nNo files to push")
            return
        
        # Push files
        print(f"\n🚀 Starting to push {len(files_to_push)} files...")
        print("="*60)
        
        success_count = 0
        failure_count = 0
        
        for filepath, message in files_to_push:
            if self.commit_and_push_file(filepath, message):
                success_count += 1
            else:
                failure_count += 1
        
        # Final summary
        print("\n" + "="*60)
        print("Push Summary:")
        print(f"  ✓ Successful: {success_count}")
        print(f"  ✗ Failed: {failure_count}")
        print("="*60)
        
        if failure_count > 0:
            print("\n⚠️  Some files failed to push. You may want to:")
            print("  1. Check your network connection")
            print("  2. Verify your git credentials")
            print("  3. Try pushing large files separately")
        else:
            print("\n✨ All files pushed successfully!")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated Git File Pusher")
    parser.add_argument('--retries', type=int, default=3, 
                        help="Maximum retry attempts for failed pushes")
    parser.add_argument('--delay', type=int, default=5,
                        help="Delay between retry attempts in seconds")
    
    args = parser.parse_args()
    
    automator = GitAutomator(max_retries=args.retries, retry_delay=args.delay)
    automator.run()

if __name__ == '__main__':
    main()