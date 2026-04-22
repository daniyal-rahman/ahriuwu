#!/bin/bash
# Prepare Windows boot with Vanguard enabled and reboot
# Run this from Linux to boot into Windows for the click-detection test

set -e

echo "============================================================================"
echo "Preparing to boot Windows with Vanguard enabled for click-detection test"
echo "============================================================================"
echo

# Check if we're on Linux
if [[ ! -f /etc/os-release ]]; then
    echo "ERROR: This script must be run on Linux (not Windows)"
    echo "Run from: ssh desktop"
    exit 1
fi

echo "Prerequisites:"
echo "  - Windows partition is available"
echo "  - Python 3 with pynput installed on Windows"
echo "  - C:\\tmp\\ directory will be created on Windows"
echo

echo "After reboot into Windows:"
echo "  1. Windows will boot with Vanguard services ENABLED"
echo "  2. Run: C:\\Repos\\ahriuwu\\scripts\\setup_windows_test.bat"
echo "  3. Run: C:\\Repos\\ahriuwu\\scripts\\test_click_detection.bat"
echo "  4. Follow the prompts to play a live game with keylogging"
echo "  5. After game: SCP replay + keylog to macOS for analysis"
echo

read -p "Ready to reboot into Windows? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo
echo "Setting up for Windows boot..."

# Create a marker file so Windows knows a test is running
touch /tmp/windows_test_marker

echo "Rebooting into Windows..."
echo "(Vanguard will be enabled for this boot)"
echo

# Try to reboot to Windows
# The exact mechanism depends on your dual-boot setup (grub, EFI, etc.)
# For UEFI/grub: modify default boot entry or use 'efibootmgr'
# For now, just trigger a standard reboot and let grub choose

sleep 2
sudo reboot
