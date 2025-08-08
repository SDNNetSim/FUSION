# pylint: disable=c-extension-no-member
# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint: disable=super-with-arguments
import os
import signal
import subprocess
import sys

from PyQt5 import QtCore



class SimulationThread(QtCore.QThread):
    """
    Sets up simulation thread runs.
    """
    progress_changed = QtCore.pyqtSignal(int)
    finished_signal = QtCore.pyqtSignal(str)
    output_hints_signal = QtCore.pyqtSignal(str)

    def __init__(self):
        super(SimulationThread, self).__init__()

        self.simulation_process = None
        self.paused = False
        self.stopped = False
        self.mutex = QtCore.QMutex()
        self.pause_condition = QtCore.QWaitCondition()

    def _run(self):
        for output_line in self.simulation_process.stdout:
            # Debug: print every output line
            # print("SimulationThread received:", output_line.strip())

            if output_line.startswith("PROGRESS:"):
                try:
                    progress_val = int(output_line.split(":", 1)[1].strip())
                    # Debug print to confirm progress was parsed:
                    #print("SimulationThread parsed progress:", progress_val)
                    self.progress_changed.emit(progress_val)
                except ValueError as e:
                    print("Error parsing progress:", e)
                continue  # Skip further processing of this line

            with QtCore.QMutexLocker(self.mutex):
                if self.stopped:
                    break
                while self.paused:
                    self.pause_condition.wait(self.mutex)
            self.output_hints_signal.emit(output_line)

        self.simulation_process.stdout.close()
        self.simulation_process.wait()

        self.finished_signal.emit('Simulation done')
        self.output_hints_signal.emit('Done...cleaning up simulation from thread')

    def run(self):
        """
        Overrides run method in QtCore.QThread.
        """
        command = os.path.join(os.getcwd(), "run_sim.py")

        with subprocess.Popen(
                args=[sys.executable, command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
        ) as process:
            self.simulation_process = process
            self._run()

    def handle_process_state(self, process_state: QtCore.QProcess.ProcessState):
        """
        Starts or runs a specific process.

        :param process_state: The current state of the process.
        :return: None
        """
        if process_state == QtCore.QProcess.ProcessState.Starting:
            self.output_hints_signal.emit('Starting process')
        elif process_state == QtCore.QProcess.ProcessState.Running:
            self.output_hints_signal.emit('Running process')

    def pause(self):
        """
        Pauses a single simulation thread.
        """
        with QtCore.QMutexLocker(self.mutex):
            os.kill(self.simulation_process.pid, signal.SIGSTOP)
            self.paused = True
            self.output_hints_signal.emit('Pausing simulation from thread')

    def resume(self):
        """
        Resumes a simulation thread.
        """
        with QtCore.QMutexLocker(self.mutex):
            os.kill(self.simulation_process.pid, signal.SIGCONT)
            self.paused = False
            self.output_hints_signal.emit('Resuming simulation from thread')
        self.pause_condition.wakeOne()

    def stop(self):
        """
        Stops a simulation thread.
        """
        with QtCore.QMutexLocker(self.mutex):
            os.kill(self.simulation_process.pid, signal.SIGKILL)
            self.stopped = True
            self.paused = False
            self.output_hints_signal.emit('Stopping simulation from thread')
        self.pause_condition.wakeOne()


