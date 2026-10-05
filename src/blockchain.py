import hashlib
import json
import time
import os
from dataclasses import dataclass
from web3 import Web3
from solcx import install_solc, compile_files

@dataclass
class Block: 
    index: int
    timestamp: float
    event: dict
    previous_hash: str
    hash: str

class AuditChain:
    """
    Hybrid Blockchain Logger:
    - Maintains a local fast-verification chain for massive telemetry and IDS logs.
    - Intercepts critical 'Break-Glass' emergency access events and transacts them
      natively to a local Ethereum (Ganache) testnet using Web3.py.
    """
    def __init__(self, rpc_url='http://127.0.0.1:8545'):
        self.blocks = []
        self._append({'type': 'genesis'})
        
        # Web3 Initialization
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.connected = self.w3.is_connected()
        self.contract = None
        self.account = None
        
        if self.connected:
            self._deploy_contract()
        else:
            print("[Web3] Notice: Ganache not detected at 127.0.0.1:8545. Falling back to local simulation.")

    def _deploy_contract(self):
        try:
            # Solcx automatically installs solc binary to ~/.solcx if missing
            install_solc('0.8.20')
            contract_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'contracts', 'BreakGlassAccess.sol'))
            
            # Compile the Solidity contract
            compiled = compile_files([contract_path], output_values=['abi', 'bin'], solc_version='0.8.20')
            contract_id, contract_interface = compiled.popitem()
            
            # Set up the default account (Doctor executing the transaction)
            self.account = self.w3.eth.accounts[0]
            self.w3.eth.default_account = self.account
            
            # Deploy the Smart Contract
            BreakGlass = self.w3.eth.contract(abi=contract_interface['abi'], bytecode=contract_interface['bin'])
            tx_hash = BreakGlass.constructor().transact({'from': self.account})
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            self.contract = self.w3.eth.contract(address=tx_receipt.contractAddress, abi=contract_interface['abi'])
            
            # Pre-license this account as a valid doctor so the Break-Glass override succeeds
            tx_hash = self.contract.functions.setLicensed(self.account, True).transact({'from': self.account})
            self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            print("[Web3] Break-Glass Smart Contract Deployed & Doctor Licensed on Ganache!")
            
        except Exception as e:
            print(f"[Web3] Warning: Failed to deploy contract: {e}")
            self.connected = False

    def _hash(self, i, t, e, p): 
        return hashlib.sha256(json.dumps({'index': i, 'timestamp': t, 'event': e, 'previous_hash': p}, sort_keys=True).encode()).hexdigest()
        
    def _append(self, e):
        p = self.blocks[-1].hash if self.blocks else '0'*64
        i = len(self.blocks)
        t = time.time()
        self.blocks.append(Block(i, t, e, p, self._hash(i, t, e, p)))
        
    def record(self, e):
        # 1. Critical Actions: Trigger Web3 Smart Contract Transaction
        if e.get('type') == 'break_glass' and self.connected and self.contract:
            try:
                # We record a hash of the commitment to prove authorization immutably
                proof_hash = hashlib.sha256(e.get('commitment', '').encode()).digest()
                tx_hash = self.contract.functions.requestEmergencyAccess(proof_hash).transact({'from': self.account})
                self.w3.eth.wait_for_transaction_receipt(tx_hash)
                
                # Attach the real blockchain transaction hash to the local event log
                e['web3_tx'] = tx_hash.hex()
                print(f"[Web3] Emergency Access Recorded Immortally on Blockchain! TxHash: {tx_hash.hex()}")
            except Exception as ex:
                print(f"[Web3] Smart Contract Transaction Failed: {ex}")
                
        # 2. Local Recording (All events)
        self._append(e)
        
    def verify(self):
        for i, b in enumerate(self.blocks):
            p = self.blocks[i-1].hash if i else '0'*64
            if b.previous_hash != p or b.hash != self._hash(b.index, b.timestamp, b.event, b.previous_hash): return False
        return True
