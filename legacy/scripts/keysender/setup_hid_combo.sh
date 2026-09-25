#!/bin/bash
# Composite HID gadget: Das-Keyboard-identity KEYBOARD (hidg0) + absolute-position
# MOUSE (hidg1) on one gadget — League needs right-click-to-move, so the keyboard-
# only rig can't play. Wireless kb+mouse combo receivers present exactly this
# composite, so the identity is unremarkable. Run on the gadget device (Pi/OTG),
# then start hid_server.py. Leaves the original setup_das_gadget.sh untouched.
set -e
G=/sys/kernel/config/usb_gadget/g1

echo "" >$G/UDC 2>/dev/null || true
rm -rf $G 2>/dev/null || true
modprobe libcomposite
mkdir -p $G && cd $G

echo 0x24F0 >idVendor            # Das Keyboard VID (matches the original rig)
echo 0x0140 >idProduct
echo 0x0200 >bcdUSB
echo 0x0100 >bcdDevice
mkdir -p strings/0x409
echo "Metadot"        >strings/0x409/manufacturer
echo "Das Keyboard"   >strings/0x409/product
echo "DK4-000001"     >strings/0x409/serialnumber

mkdir -p configs/c.1/strings/0x409
echo "kb+mouse" >configs/c.1/strings/0x409/configuration
echo 250 >configs/c.1/MaxPower

# --- function 0: boot keyboard (8-byte reports, same as the original rig) ---
mkdir -p functions/hid.usb0
echo 1 >functions/hid.usb0/protocol
echo 1 >functions/hid.usb0/subclass
echo 8 >functions/hid.usb0/report_length
echo -ne \
'\x05\x01\x09\x06\xa1\x01\x05\x07\x19\xe0\x29\xe7\x15\x00\x25\x01\x75\x01\x95\x08\x81\x02\x95\x01\x75\x08\x81\x03\x95\x05\x75\x01\x05\x08\x19\x01\x29\x05\x91\x02\x95\x01\x75\x03\x91\x03\x95\x06\x75\x08\x15\x00\x25\x65\x05\x07\x19\x00\x29\x65\x81\x00\xc0' \
>functions/hid.usb0/report_desc

# --- function 1: mouse -------------------------------------------------------
# TWO descriptors, and picking the wrong one is a SILENT failure: the report
# length is baked into the descriptor, so a sender writing 4-byte relative
# reports into a 6-byte absolute gadget (or vice versa) gets EINVAL or garbage,
# with nothing in the game to say so.
#
# MOUSE_MODE=rel (DEFAULT) — 4-byte relative reports [buttons, dx, dy, wheel],
#   signed -127..127. This is what the rig runs and what hybrid_sender.py's
#   corner-relative addressing speaks. A relative pointer cannot be asked where
#   the cursor is, which is why the sender re-zeros against a screen corner.
# MOUSE_MODE=abs — 6-byte absolute reports [buttons, x16, y16, wheel], logical
#   0..32767 across the pointer's target surface. Simpler in principle (no
#   position tracking) but the surface it addresses is not knowable without
#   measuring it on the host, so the rig does not use it.
#
#     MOUSE_MODE=abs ./setup_hid_combo.sh    # only if you know you want this
MOUSE_MODE="${MOUSE_MODE:-rel}"
mkdir -p functions/hid.usb1
echo 2 >functions/hid.usb1/protocol
echo 1 >functions/hid.usb1/subclass
if [ "$MOUSE_MODE" = "abs" ]; then
  echo 6 >functions/hid.usb1/report_length
  echo -ne \
'\x05\x01\x09\x02\xa1\x01\x09\x01\xa1\x00\x05\x09\x19\x01\x29\x03\x15\x00\x25\x01\x95\x03\x75\x01\x81\x02\x95\x01\x75\x05\x81\x03\x05\x01\x09\x30\x09\x31\x16\x00\x00\x26\xff\x7f\x75\x10\x95\x02\x81\x02\x09\x38\x15\x81\x25\x7f\x75\x08\x95\x01\x81\x06\xc0\xc0' \
  >functions/hid.usb1/report_desc
else
  echo 4 >functions/hid.usb1/report_length
  echo -ne \
'\x05\x01\x09\x02\xa1\x01\x09\x01\xa1\x00\x05\x09\x19\x01\x29\x03\x15\x00\x25\x01\x95\x03\x75\x01\x81\x02\x95\x01\x75\x05\x81\x03\x05\x01\x09\x30\x09\x31\x09\x38\x15\x81\x25\x7f\x75\x08\x95\x03\x81\x06\xc0\xc0' \
  >functions/hid.usb1/report_desc
fi

ln -s functions/hid.usb0 configs/c.1/
ln -s functions/hid.usb1 configs/c.1/

UDC=$(ls /sys/class/udc | head -1)
echo "$UDC" >UDC
echo "HID combo gadget up on UDC $UDC: /dev/hidg0 (keyboard), /dev/hidg1 (${MOUSE_MODE} mouse, $(cat functions/hid.usb1/report_length)-byte reports)"
