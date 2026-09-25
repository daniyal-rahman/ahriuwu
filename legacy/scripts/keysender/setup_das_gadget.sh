#!/bin/bash
set -e

echo "Setting up Das Keyboard-compatible HID gadget..."

# Clean any previous gadget
echo "" >/sys/kernel/config/usb_gadget/g1/UDC 2>/dev/null || true
rm -rf /sys/kernel/config/usb_gadget/g1 2>/dev/null || true

modprobe libcomposite

cd /sys/kernel/config/usb_gadget/
mkdir -p g1
cd g1

# === Real Das Keyboard identity ===
echo "0x24F0" >idVendor  # Das Keyboard VID
echo "0x0140" >idProduct # Your exact PID
echo "0x0200" >bcdUSB
echo "0x0100" >bcdDevice

mkdir -p strings/0x409
echo "Das Keyboard" >strings/0x409/manufacturer
echo "Das Keyboard" >strings/0x409/product
echo "DK4QXXXXXX" >strings/0x409/serialnumber # realistic-looking serial

# HID function
mkdir -p functions/hid.usb0
echo 1 >functions/hid.usb0/protocol # Keyboard
echo 1 >functions/hid.usb0/subclass
echo 8 >functions/hid.usb0/report_length

# Standard boot-protocol compatible keyboard descriptor
printf '\x05\x01\x09\x06\xa1\x01\x05\x07\x19\xe0\x29\xe7\x15\x00\x25\x01\x75\x01\x95\x08\x81\x02\x75\x08\x95\x01\x81\x03\x95\x05\x75\x01\x05\x08\x19\x01\x29\x05\x91\x02\x95\x01\x75\x03\x91\x03\x95\x06\x75\x08\x15\x00\x25\x65\x05\x07\x19\x00\x29\x65\x81\x00\xc0' >functions/hid.usb0/report_desc

# Configuration
mkdir -p configs/c.1/strings/0x409
echo "Das Keyboard Config" >configs/c.1/strings/0x409/configuration
echo 250 >configs/c.1/MaxPower
ln -s functions/hid.usb0 configs/c.1/

# Bind to UDC (change this to your actual controller)
# Run: ls /sys/class/udc/
echo "CHANGE_THIS_TO_YOUR_UDC" >UDC

echo "Gadget ready. /dev/hidg0 should now exist."
ls -l /dev/hidg0
