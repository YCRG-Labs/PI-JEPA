import torch
import math


class EMATeacher:
    def __init__(
        self,
        student,
        teacher,
        tau_start=0.996,
        tau_end=0.9995,
        total_steps=100000
    ):
        self.student = student
        self.teacher = teacher

        self.tau_start = tau_start
        self.tau_end = tau_end
        self.total_steps = total_steps

        self.step = 0

        self._init_teacher()

    def _init_teacher(self):
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.data.copy_(ps.data)
            pt.requires_grad = False

        for bs, bt in zip(self.student.buffers(), self.teacher.buffers()):
            bt.data.copy_(bs.data)

    def _tau(self):
        t = min(self.step / self.total_steps, 1.0)
        return self.tau_start + (self.tau_end - self.tau_start) * (1 - math.cos(math.pi * t)) / 2

    @torch.no_grad()
    def update(self):
        tau = self._tau()

        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.data.mul_(tau).add_(ps.data * (1.0 - tau))

        for bs, bt in zip(self.student.buffers(), self.teacher.buffers()):
            bt.data.copy_(bs.data)

        self.step += 1

    def state_dict(self):
        return {
            "teacher": self.teacher.state_dict(),
            "step": self.step
        }

    def load_state_dict(self, state):
        self.teacher.load_state_dict(state["teacher"])
        self.step = state["step"]
