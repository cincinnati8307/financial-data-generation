#!/usr/bin/env python3
"""
Simple Git Push Automation Script
Automates pushing files one by one with error handling and retry logic.

Usage:
    python simple_push.py                    # Interactive mode
    python simple_push.py --all              # Push all untracked files
    python simple_push.py --pattern "*.md"   # Push files matching pattern
    python simple_push.py --skip-large       # Skip files >1000 lines
"""

import os
import subprocess
import time
from pathlib import Path
from typing import List, Tuple

class SimpleGitPusher:
    def __init__(self, max_retries: int = 3, retry_delay: int = 5, 
                 large_threshold: int = 1000):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.large_threshold = large_threshold
    
    def run_cmd(self, cmd: List[str], check: bool = True) -> Tuple[bool, str]:
        """Run command and return (success, output)"""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=check)
            return True, result.stdout
        except Exception as e:
            return False, str(e)
    
    def get_line_count(self, filepath: Path) -> int:
        """Count lines in file"""
        try:
            return sum(1 for _ in open(filepath, encoding='utf-8', errors='ignore'))
        except:
            return 0
    
    def promote_file(self, filepath: Path, message: str) -> bool:
        """Push a single file with retry logic"""
        lines = self.get_line_count(filepath)
        is_large = lines > self.large_threshold
        
        print(f"\n{'='*50}")
        print(f"📤 {filepath.name}")
        print(f"   Size: {lines:,} lines ({'Large' if is_large else 'Small'})")
        print(f"   Message: {message}")
        
        if is_large:
            print(f"   ⚠️  Large file - may need retries")
        
        for attempt in range(1, self.max_retries + 1):
            print(f"   Attempt {attempt}/{self.max_retries}...", end=" ")
            
            try:
                # Add
                if not self.run_cmd(['git', 'add', str(filepath)])[0]:
                    print("❌ Add failed")
                    return False
                
                # Commit
                commit_msg = f"{message} | {lines:,} lines"
                success, out = self.run_cmd(['git', 'commit', '-m', commit_msg])
                if 'nothing to commit' in out.lower():
                    print("✓ Already committed")
                    return True
                elif not success:
                    print("❌ Commit failed")
                    return False
                
                # Push
                success, out = self.run_cmd(['git', 'push'])
                if success:
                    print("✅ Success!")
                    return True
                else:
                    error = out.lower()
                    if '403' in error:
                        print("❌ Auth error")
                    elif 'rpc failed' in error or 'network' in error:
                        print("❌ Network error, retrying...")
                        if attempt < self.max_retries:
                            time.sleep(self.retry_delay)
                            continue
                    else:
                        print(f"❌ Push failed: {error[:50]}")
                
                # Reset for retry
                self.run_cmd(['git', 'reset', '--hard', 'HEAD~1'], check=False)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                self.run_cmd(['git', 'reset', '--hard', 'HEAD~1'], check=False)
            
            if attempt < self.max_retries:
                time.sleep(self.retry_delay)
        
        print(f"❌ Failed after {self.max_retries} attempts")
        return False
    
    def get_untracked_files(self) -> List[Path]:
        """Get list of untracked files"""
        success, output = self.run_cmd(['git', 'status', '--short'])
        files = []
        
        if success:
            for line in output.split('\n'):
                if line.startswith('??'):
                    path = Path(line.strip()[3:].strip())
                    if path.exists() and not path.is_dir():
                        files.append(path)
        
        return sorted(files)
    
    def push_all(self, skip_large: bool = False) -> Tuple[int, int]:
        """Push all untracked files
        
        Returns: (success_count, failure_count)
        """
        files = self.get_untracked_files()
        if not files:
            print("No untracked files found")
            return 0, 0
        
        print(f"Found {len(files)} untracked files")
        
        success = 0
        failure = 0
        
        for filepath in files:
            lines = self.get_line_count(filepath)
            
            if skip_large and lines > self.large_threshold:
                print(f"⏭️  Skipping large file: {filepath.name} ({lines:,} lines)")
                continue
            
            message = f"Add {filepath.name}"
            if self.promote_file(filepath, message):
                success += 1
            else:
                failure += 1
        
        return success, failure
    
    def push_pattern(self, pattern: str, skip_large: bool = False) -> Tuple[int, int]:
        """Push files matching glob pattern
        
        Returns: (success_count, failure_count)
        """
        files = sorted(Path('.').glob(pattern))
        files = [f for f in files if f.exists() and not f.is_dir()]
        
        if not files:
            print(f"No files found matching '{pattern}'")
            return 0, 0
        
        print(f"Found {len(files)} files matching '{pattern}'")
        
        success = 0
        failure = 0
        
        for filepath in files:
            lines = self.get_line_count(filepath)
            
            if skip_large and lines > self.large_threshold:
                print(f"⏭️  Skipping large file: {filepath.name} ({lines:,} lines)")
                continue
            
            message = f"Add {filepath.name}"
            if self.promote_file(filepath, message):
                success += 1
            else:
                failure += 1
        
        return success, failure

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Simple Git File Push Automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simple_push.py                    # Interactive mode
  python simple_push.py --all              # Push all untracked files  
  python simple_push.py --pattern "*.md"   # Push all markdown files
  python simple_push.py --skip-large       # Skip files >1000 lines
  python simple_push.py --all --skip-large # Push but skip large files
        """
    )
    
    parser.add_argument('--all', action='store_true', help='Push all untracked files')
    parser.add_argument('--pattern', type=str, help='Glob pattern for files to push')
    parser.add_argument('--skip-large', action='store_true', 
                       help=f'Skip files larger than {1000} lines')
    parser.add_argument('--retries', type=int, default=3, 
                       help='Max retry attempts (default: 3)')
    parser.add_argument('--delay', type=int, default=5,
                       help='Delay between retries in seconds (default: 5)')
    parser.add_argument('--threshold', type=int, default=1000,
                       help='Large file threshold in lines (default: 1000)')
    
    args = parser.parse_args()
    
    print("🚀 Simple Git Push Automation")
    print("="*50)
    
    pusher = SimpleGitPusher(
        max_retries=args.retries, 
        retry_delay=args.delay,
        large_threshold=args.threshold
    )
    
    success_count = 0
    failure_count = 0
    
    if args.all:
        # Push all untracked files
        success_count, failure_count = pusher.push_all(skip_large=args.skip_large)
        
    elif args.pattern:
        # Push files matching pattern
        success_count, failure_count = pusher.push_pattern(args.pattern, skip_large=args.skip_large)
        
    else:
        # Interactive mode
        files = pusher.get_untracked_files()
        if not files:
            print("No untracked files found")
            return
        
        print(f"\nFound {len(files)} untracked files:")
        for i, f in enumerate(files, 1):
            lines = pusher.get_line_count(f)
            size_icon = "📊" if lines > args.threshold else "📝"
            print(f"  {i}. {f} ({lines:,} lines) {size_icon}")
        
        print("\nChoose option:")
        print("  1. Push all files")
        print("  2. Push files by number (comma-separated)")
        print("  3. Skip large files only")
        print("  q. Quit")
        
        choice = input("\n> ").strip().lower()
        
        if choice == 'q':
            return
        elif choice == '1':
            success_count, failure_count = pusher.push_all()
        elif choice == '2':
            nums = input("Enter file numbers (comma-separated): ").strip()
            try:
                indices = [int(x) for x in nums.split(',') if x.strip().isdigit()]
                total = 0
                for i in indices:
                    if 1 <= i <= len(files):
                        filepath = files[i-1]
                        if pusher.promote_file(filepath, f"Add {filepath.name}"):
                            success_count += 1
                        else:
                            failure_count += 1
                        total += 1
                if total == 0:
                    print("No valid files selected")
            except ValueError:
                print("Invalid input")
        elif choice == '3':
            success_count, failure_count = pusher.push_all(skip_large=True)
        else:
            print("Invalid choice")
            return
    
    # Summary
    print("\n" + "="*50)
    print("📊 Summary:")
    print(f"  ✅ Successful: {success_count}")
    print(f"  ❌ Failed: {failure_count}")
    print("="*50)
    
    if failure_count > 0:
        print("\n⚠️  Some files failed. Consider:")
        print("  • Check network connection")
        print("  • Verify git credentials")
        print("  • Try large files separately with --skip-large")
    else:
        print("\n✨ All files pushed successfully!")

if __name__ == '__main__':
    main()