#!/usr/bin/python
# Note: This script requires a Linux VM with Mininet-WiFi installed.
# It simulates the 6G network slices for the Bio-Twin prototype.

from mininet.log import setLogLevel, info
from mn_wifi.cli import CLI
from mn_wifi.net import Mininet_wifi
from mn_wifi.link import wmediumd
from mn_wifi.wmediumdConnector import interference

def create_6g_topology():
    info("*** Creating 6G Network Simulation (Mininet-WiFi) ***\n")
    net = Mininet_wifi(link=wmediumd, wmediumd_mode=interference)

    info("*** Adding stations (IoMT Devices & Surgeon Consoles) ***\n")
    # Patient IoMT Node
    iomt = net.addStation('iomt1', mac='00:00:00:00:00:01', ip='10.0.0.1/8', position='10,20,0')
    # Surgeon Console
    surgeon = net.addStation('surg1', mac='00:00:00:00:00:02', ip='10.0.0.2/8', position='50,20,0')
    # Edge Hospital Server
    hospital = net.addHost('hosp1', ip='10.0.0.3/8')

    info("*** Adding 6G Access Points (gNodeB) ***\n")
    ap1 = net.addAccessPoint('ap1', ssid='6G-Core', mode='ac', channel='36', position='30,20,0')
    c0 = net.addController('c0')

    info("*** Configuring Network Slices via Traffic Control (TC) ***\n")
    # In a real 6G deployment, this would be done via SDN/OpenFlow rules. 
    # Here we mock URLLC (Ultra-Reliable Low-Latency) for Tele-surgery, 
    # and eMBB for massive IoMT data.
    net.configureWifiNodes()
    
    # Slice 1: URLLC (Tele-surgery) -> Extremely low latency, high priority
    net.addLink(surgeon, ap1, bw=1000, delay='1ms', loss=0.01)
    
    # Slice 2: eMBB (Bio-Twin Telemetry) -> Standard latency, high bandwidth
    net.addLink(iomt, ap1, bw=500, delay='15ms', loss=1.0)
    
    # Edge Link
    net.addLink(ap1, hospital, bw=10000, delay='0.5ms')

    info("*** Starting network ***\n")
    net.build()
    c0.start()
    ap1.start([c0])

    info("*** Running Tele-Surgery Ping Test over URLLC Slice ***\n")
    net.ping([surgeon, hospital])

    info("*** Starting CLI ***\n")
    # Run the interactive CLI so researchers can test packet injection attacks
    CLI(net)

    info("*** Stopping network ***\n")
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    create_6g_topology()
