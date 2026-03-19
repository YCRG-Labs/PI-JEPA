import torch
import math


class EMATeacher:
    def __init__(
        self,
        student,
        teacher,
        tau_start=0.996,
        tau_end=0.9995,
        total_steps=100000,
        update_every=1,      
    ):
        self.student = student
        self.teacher = teacher

        self.tau_start = tau_start
        self.tau_end = tau_end
        self.total_steps = total_steps
        self.update_every = update_every

        self.step = 0

        self._init_teacher()

    def _init_teacher(self):
        # copy params
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.detach().copy_(ps.detach())
            pt.requires_grad_(False)

        # copy buffers (e.g. BN stats)
        for bs, bt in zip(self.student.buffers(), self.teacher.buffers()):
            bt.copy_(bs)

        self.teacher.eval()


    def _tau(self):
        t = min(self.step / self.total_steps, 1.0)

        return self.tau_start + (self.tau_end - self.tau_start) * (
            (1 - math.cos(math.pi * t)) / 2
        )

    @torch.no_grad()
    def update(self):
        if self.step % self.update_every != 0:
            self.step += 1
            return

        tau = self._tau()

        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            if ps.requires_grad:
                pt.lerp_(ps, 1.0 - tau)  

        for bs, bt in zip(self.student.buffers(), self.teacher.buffers()):
            bt.copy_(bs)

        self.step += 1


    def state_dict(self):
        return {
            "teacher": self.teacher.state_dict(),
            "step": self.step
        }

    def load_state_dict(self, state):
        self.teacher.load_state_dict(state["teacher"])
        self.step = state["step"]


@torch.no_grad()
def update_ema(student, teacher, tau):
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        if ps.requires_grad:
            pt.lerp_(ps, 1.0 - tau)

    for bs, bt in zip(student.buffers(), teacher.buffers()):
        bt.copy_(bs)