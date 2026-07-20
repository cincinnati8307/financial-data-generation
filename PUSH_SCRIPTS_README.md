# Git Automation Scripts

This directory contains automated scripts for pushing files to GitHub individually with error handling and retry logic.

## Available Scripts

### 1. `simple_push.py` - Recommended for general use

A simple, easy-to-use script for automating git file pushes with built-in retry logic for large files and network issues.

#### Usage

```bash
# Interactive mode
python simple_push.py

# Push all untracked files
python simple_push.py --all

# Push files matching a pattern
python simple_push.py --pattern "*.md"

# Push all but skip large files (>1000 lines)
python simple_push.py --all --skip-large

# Custom settings
python simple_push.py --all --retries 5 --delay 10 --threshold 2000
```

#### Features

- ✅ **Interactive mode** for selective file pushing
- ✅ **Automatic retry** for network failures (up to 3 attempts by default)
- ✅ **Large file detection** and optional skipping
- ✅ **Progress feedback** with clear status messages
- ✅ **Error recovery** with automatic reset on failures
- ✅ **Configurable settings** for retries, delays, and thresholds

#### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--all` | Push all untracked files | - |
| `--pattern` | Push files matching glob pattern | - |
| `--skip-large` | Skip files > threshold (lines) | - |
| `--retries` | Max retry attempts | 3 |
| `--delay` | Delay between retries (seconds) | 5 |
| `--threshold` | Large file threshold (lines) | 1000 |

### 2. `push_files.py` - Advanced automation

A more advanced version with additional features for complex workflows.

#### Usage

```bash
# Interactive selection
python push_files.py

# Custom configuration
python push_files.py --retries 5 --delay 10
```

#### Features

- ✅ Interactive file selection with numbered list
- ✅ Git configuration validation
- ✅ Detailed error diagnostics
- ✅ File size analysis and warnings
- ✅ Comprehensive logging and progress tracking
- ✅ Support for file patterns and bulk operations

#### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--retries` | Maximum retry attempts | 3 |
| `--delay` | Delay between retries (seconds) | 5 |

## Use Cases

### 1. Pushing All Documentation Files

```bash
python simple_push.py --pattern "*.md"
```

### 2. Pushing All Newly Created Files

```bash
python simple_push.py --all
```

### 3. Careful Pushing (Skip Large Files First)

```bash
# First push small files
python simple_push.py --all --skip-large

# Then push large files manually one by one
python simple_push.py --pattern "results/query_aware_comparison/predictions.jsonl"
python simple_push.py --pattern "results/query_aware_comparison/plots/*.png"
```

### 4. Interactive Selection

```bash
python simple_push.py
```

This will show you all untracked files and let you choose which to push interactively.

### 5. Pushing Specific File Types

```bash
# Push just source code
python simple_push.py --pattern "src/**/*.py"

# Push just result files
python simple_push.py --pattern "results/**/*.json"

# Push specific directory
python simple_push.py --pattern "docs/*"
```

## How It Works

### Automatic File Detection

The scripts automatically detect untracked files using `git status` and display them with:
- File size (line count)
- Size category (Small 📝 / Large 📊)
- Relative path

### Retry Logic

When a push fails, the scripts:
1. Detect the type of error (network, auth, etc.)
2. Automatically reset to the previous state
3. Wait for the configured delay
4. Retry the push up to the maximum attempts
5. Skip to the next file if all retries fail

### Large File Handling

Large files (by default >1,000 lines) are:
- Clearly identified when detected
- Can be automatically skipped with `--skip-large`
- Recommended for manual, one-at-a-time pushing

### Error Recovery

If a push fails:
- The script automatically resets: `git reset --hard HEAD~1`
- Leaves the repository in a clean state
- Continues with the next file
- Provides clear error messages

## Example Workflows

### Initial Setup

```bash
# Check git configuration
git remote get-url origin

# Ensure credentials are configured
# URL should look like: https://TOKEN@github.com/USER/REPO
```

### Workflow 1: Careful Documentation Push

```bash
# Push documentation first (usually smaller files)
python simple_push.py --pattern "*.md"

# Push source code
python simple_push.py --pattern "src/**/*.py"

# Skip large results files
python simple_push.py --all --skip-large
```

### Workflow 2: Full Automated Push

```bash
# Attempt everything with extended retries
python simple_push.py --all --retries 5 --delay 10
```

### Workflow 3: Manual Control

```bash
# Run interactive mode
python simple_push.py

# Review all files
# Choose option 1 (all) or 2 (select specific files)
# Watch the progress
# Review summary
```

### Workflow 4: Problematic Files Recovery

```bash
# First push everything that will work
python simple_push.py --all --skip-large

# Then push remaining files individually
for file in results/query_aware_comparison/plots/*.png; do
    python simple_push.py --pattern "$file"
done
```

## Troubleshooting

### Authentication Errors (403)

```bash
# Update remote URL with token
git remote set-url origin https://TOKEN@github.com/USER/REPO

# Verify configuration
git remote get-url origin
```

### Network Errors

- The script will automatically retry
- Increase `--delay` between retries
- Try pushing large files separately
- Check your internet connection

### Large Files Still Failing

```bash
# Increase threshold or reduce file size
python simple_push.py --all --threshold 5000

# Or handle manually
git add path/to/large/file
git commit -m "Add large file"
git push  # Try manually
```

### Repository Not Clean

```bash
# Reset to clean state
git reset --hard HEAD
git clean -fd

# Start fresh with script
python simple_push.py --all
```

## Best Practices

1. **Start Small**: Begin with small files to verify configuration
2. **Use Interactive Mode**: For anything complex, use interactive selection
3. **Skip Large Files**: First push small files, then handle large ones individually
4. **Monitor Progress**: Watch the output and retry messages
5. **Check Configuration**: Ensure git credentials are properly set
6. **Handle Failures Manually**: If a file consistently fails, push it manually

## Configuration

### Environment Variables

You can set environment variables for convenience:

```bash
# Set custom defaults
export GIT_PUSH_RETRIES=5
export GIT_PUSH_DELAY=10
export GIT_PUSH_THRESHOLD=2000
```

### Git Remote Configuration

Ensure your remote URL includes your credentials:

```bash
# Check current remote
git remote get-url origin

# Update with token
git remote set-url origin https://YOUR_TOKEN@github.com/USER/REPO

# Store credential helper
git config --global credential.helper store
```

## Performance Considerations

- **Small files (<1KB)**: Push instantly
- **Medium files (1-10KB)**: Usually succeed on first try
- **Large files (10-100KB)**: May require retries
- **Very large files (>100KB)**: Push manually, use Git LFS

## Integration with CI/CD

You can use `simple_push.py` in CI/CD workflows:

```bash
# In your CI script
pip install -r requirements.txt  # if needed

# Push results
python simple_push.py --all --skip-large

echo "Push completed successfully"
```

## Contributing

When adding new automation features:

1. Keep it simple and focused
2. Add clear error messages
3. Support both interactive and automated modes
4. Provide good progress feedback
5. Handle edge cases gracefully

## License

These scripts are part of the Sensitive Egress Privacy Detection Framework project.