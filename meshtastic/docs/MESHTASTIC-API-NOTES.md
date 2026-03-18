# Meshtastic Python Library API Notes

Hard-won knowledge from real hardware testing with the T-LoRa Pager.

## Metadata Field Names are camelCase

The `DeviceMetadata` protobuf uses camelCase, NOT snake_case:

```python
# WRONG:
metadata.has_wifi      # returns AttributeError or default
metadata.has_bluetooth

# CORRECT:
metadata.hasWifi       # True/False
metadata.hasBluetooth  # True/False
metadata.hasEthernet
metadata.canShutdown
metadata.hasPKC
metadata.firmware_version  # This one IS snake_case (exception)
```

## localConfig is on localNode, NOT on the interface

```python
# WRONG — always None with noNodes=True:
iface.localConfig

# CORRECT:
iface.localNode.localConfig.lora.region        # int (1=US)
iface.localNode.localConfig.lora.modem_preset  # int (0=LONG_FAST)
iface.localNode.localConfig.lora.tx_power      # int (dBm)
iface.localNode.localConfig.lora.hop_limit     # int
iface.localNode.localConfig.bluetooth.enabled  # bool
iface.localNode.localConfig.network.wifi_enabled  # bool
```

## Enum Conversions

Region and modem_preset are protobuf enum integers:

```python
from meshtastic.protobuf.config_pb2 import Config

# Region: 0=UNSET, 1=US, 2=EU_433, 3=EU_868, etc.
Config.LoRaConfig.RegionCode.Name(1)  # "US"

# Modem: 0=LONG_FAST, 4=MEDIUM_FAST, etc.
Config.LoRaConfig.ModemPreset.Name(4)  # "MEDIUM_FAST"

# Channel role: 0=DISABLED, 1=PRIMARY, 2=SECONDARY
from meshtastic.protobuf.channel_pb2 import Channel
Channel.Role.Name(1)  # "PRIMARY"
```

## noNodes=True Behavior

- `noNodes=True` skips receiving the node database (fast connect ~5s vs ~60s for 250 nodes)
- Config exchange STILL happens — `localNode.localConfig` IS populated
- `metadata` IS populated from the config exchange
- Node list (`iface.nodes`) will be empty
- `iface.localConfig` (top-level) is None — use `iface.localNode.localConfig`

## Serial Port Recovery

After rapid connect/disconnect cycles, the serial buffer accumulates stale protobuf data.
The meshtastic library's protocol handshake fails because it reads stale data first.

Fix: drain the serial buffer and toggle DTR before connecting:

```python
import serial, time
s = serial.Serial(port, 115200, timeout=0.5)
s.dtr = False; time.sleep(0.1); s.dtr = True; time.sleep(0.5)
while s.in_waiting:
    s.read(s.in_waiting)
    time.sleep(0.05)
s.close()
```

## T-LoRa Pager Specifics

- ESP32-S3 with native USB (VID 303a, PID 1001)
- Appears as `/dev/ttyACM0` (CDC-ACM, not CP210x/CH340)
- BLE name: `Meshtastic_ff38` (last 4 hex of MAC)
- 250 nodes in Bay Area mesh, full connect takes 60+ seconds
- WiFi + BLE available, GPS via U-blox MIA-M10Q
- Default modem should be MEDIUM_FAST for Bay Area

## Known BLE Issues on Linux

- `meshtastic.ble_interface.BLEInterface` uses its own scanner that doesn't keep devices in bluez cache
- Direct `bleak.BleakClient` connection fails with "device not found" unless scanner is kept active
- Workaround: use `BleakScanner` to discover, then connect immediately with the device object (not address)
- BLE pairing requires PIN (default 123456, configurable via USB)
