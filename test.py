import asyncio
import sys

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from ib_insync import *

ib = IB()
ib.connect('127.0.0.1', 4002, clientId=1)

contract = Forex('XAUUSD')
ib.qualifyContracts(contract)

ticker = ib.reqMktData(contract)
ib.sleep(2)
print("Prix XAUUSD:", ticker.last)