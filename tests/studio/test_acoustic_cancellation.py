"""Obsolete acoustic UI work is skipped even after entering native queues."""
import asyncio
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from studio.core.utils.perception.component import Perception
from voicemem.engine import BackgroundCapacity


class AcousticCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_or_owner_change_skips_capacity_queued_model(self):
        for cancel_task in (False, True):
            with self.subTest(cancel_task=cancel_task):
                capacity = BackgroundCapacity(enabled=True)
                held, release = threading.Event(), threading.Event()

                def occupy():
                    with capacity.reserve('cpu'):
                        held.set()
                        release.wait(3)

                worker = threading.Thread(target=occupy)
                worker.start()
                self.assertTrue(held.wait(1))
                current = [True]
                agent = Perception()
                generate = Mock(return_value=[dict(labels=['neutral'], scores=[1.0])])
                agent._E2V = {'m': SimpleNamespace(kwargs={'device': 'cpu'}, generate=generate)}
                agent._E2V_MAP = {'neutral': 'neutral'}
                agent.BARGE_DEBUG = False
                agent.ACOUSTIC_MIN_SCORE = .5
                agent.ACOUSTIC_TRUST = {'neutral'}
                agent.wait_idle = AsyncMock()
                send = AsyncMock()
                with patch('voicemem.engine.telemetry.metrics.background_capacity', capacity):
                    task = agent._kick_acoustic(send, 'synthetic-unused.wav', output_id='synthetic',
                                                is_current=lambda: current[0])
                    try:
                        async with asyncio.timeout(1):
                            while capacity.snapshot()['waiting'] != 1:
                                await asyncio.sleep(.001)
                        if cancel_task:
                            task.cancel()
                        else:
                            current[0] = False
                        await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 1)
                        await asyncio.wait_for(asyncio.gather(*tuple(agent._BACKGROUND_LANE.tasks)), 1)
                        self.assertEqual(capacity.snapshot()['waiting'], 0)
                        self.assertEqual(capacity.snapshot()['active'], 1)
                    finally:
                        release.set()
                        worker.join(1)
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                generate.assert_not_called()
                send.assert_not_awaited()
