# -*- coding: utf-8 -*-
'''
Гауссов импульс распространяется в одну сторону (TFSF boundary).
Источник находится в диэлектрике.
'''

import numpy
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt
import tools

class Ricker:
    '''
    Источник, создающий импульс в форме вейвлета Рикера
    '''

    def __init__(self, Np, Md, eps=1.0, mu=1.0, Sc=1.0, magnitude=1.0):
        '''
        magnitude - максимальное значение в источнике;
        Nl - количество отсчетов на длину волны;
        Md - определяет задержку импульса;
        Sc - число Куранта;
        magnitude - максимальное значение в источнике.
        '''
        self.Np = Np
        self.Md = Md
        self.eps = eps
        self.mu = mu
        self.Sc = Sc
        self.magnitude = magnitude

    def getField(self, m, q):
        t = (numpy.pi ** 2) * (self.Sc *
            (q - m * numpy.sqrt(self.eps * self.mu)) / self.Np - self.Md) ** 2
        return self.magnitude * (1 - 2 * t) * numpy.exp(-t)
    

if __name__ == '__main__':
    # Волновое сопротивление свободного пространства
    W0 = 120.0 * numpy.pi

    # Число Куранта
    Sc = 1.0

    # Скорость света
    c = 3e8

    # Время расчета в отсчетах
    maxTime = 400

    # Размер области моделирования вдоль оси X в метрах
    X = 2.0

    #Размер ячейки разбиения
    dx = 1e-2

    # Размер области моделирования в отсчетах
    maxSize = int(X / dx)

    #Шаг дискретизации по времени
    dt = Sc * dx / c

    # Положение источника в отсчетах
    sourcePos = int(maxSize / 2)

    # Датчики для регистрации поля
    probesPos = [sourcePos + 75]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    # Параметры среды
    # Диэлектрическая проницаемость
    eps = numpy.ones(maxSize)
    eps[:] = 5.0

    # Магнитная проницаемость
    mu = numpy.ones(maxSize - 1)

    Ez = numpy.zeros(maxSize)
    Hy = numpy.zeros(maxSize - 1)

    source = Ricker(30.0, 1.5, eps[sourcePos], mu[sourcePos])

    # Коэффициенты для расчета ABC второй степени
    # Sc' для левой границы
    Sc1Left = Sc / numpy.sqrt(mu[0] * eps[0])

    k1Left = -1 / (1 / Sc1Left + 2 + Sc1Left)
    k2Left = 1 / Sc1Left - 2 + Sc1Left
    k3Left = 2 * (Sc1Left - 1 / Sc1Left)
    k4Left = 4 * (1 / Sc1Left + Sc1Left)

    # Sc' для правой границы
    Sc1Right = Sc / numpy.sqrt(mu[-1] * eps[-1])

    k1Right = -1 / (1 / Sc1Right + 2 + Sc1Right)
    k2Right = 1 / Sc1Right - 2 + Sc1Right
    k3Right = 2 * (Sc1Right - 1 / Sc1Right)
    k4Right = 4 * (1 / Sc1Right + Sc1Right)

    # Ez[0: 2] в предыдущий момент времени (q)
    oldEzLeft1 = numpy.zeros(3)

    # Ez[0: 2] в пред-предыдущий момент времени (q - 1)
    oldEzLeft2 = numpy.zeros(3)

    # Ez[-3: -1] в предыдущий момент времени (q)
    oldEzRight1 = numpy.zeros(3)

    # Ez[-3: -1] в пред-предыдущий момент времени (q - 1)
    oldEzRight2 = numpy.zeros(3)
    
    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -1.1
    display_ymax = 1.1

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel, dx)

    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])

    for q in range(maxTime):
        
        # Расчет компоненты поля H
        Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] -= Sc / (W0 * mu[sourcePos - 1]) * source.getField(0, q)

        # Расчет компоненты поля E
        Hy_shift = Hy[:-1]
        Ez[1:-1] = Ez[1:-1] + (Hy[1:] - Hy_shift) * Sc * W0 / eps[1:-1]

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] += (Sc / (numpy.sqrt(eps[sourcePos] * mu[sourcePos])) *
                          source.getField(-0.5, q + 0.5))

        # Граничные условия ABC второй степени (слева)
        Ez[0] = (k1Left * (k2Left * (Ez[2] + oldEzLeft2[0]) +
                           k3Left * (oldEzLeft1[0] + oldEzLeft1[2] - Ez[1] - oldEzLeft2[1]) -
                           k4Left * oldEzLeft1[1]) - oldEzLeft2[2])

        oldEzLeft2[:] = oldEzLeft1[:]
        oldEzLeft1[:] = Ez[0: 3]

        # Граничные условия ABC второй степени (справа)
        Ez[-1] = (k1Right * (k2Right * (Ez[-3] + oldEzRight2[-1]) +
                             k3Right * (oldEzRight1[-1] + oldEzRight1[-3] - Ez[-2] - oldEzRight2[-2]) -
                             k4Right * oldEzRight1[-2]) - oldEzRight2[-3])

        oldEzRight2[:] = oldEzRight1[:]
        oldEzRight1[:] = Ez[-3:]

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if q % 5 == 0:
            display.updateData(display_field, q)

    display.stop()

    # Отображение сигнала, сохраненного в датчиках
    tools.showProbeSignals(probes, -1.1, 1.1, dt)

    # Отображение спектра сигнала
    tools.Spectrum(probe.E, dt)

