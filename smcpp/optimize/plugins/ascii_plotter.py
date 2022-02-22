import numpy as np
import shutil
import subprocess
import tempfile
from logging import getLogger

from .optimizer_plugin import OptimizerPlugin, targets
import smcpp.defaults

logger = getLogger(__name__)


class AsciiPlotter(OptimizerPlugin):

    def __init__(self):
        self._gnuplot_path = shutil.which("gnuplot")

    @targets(["post M-step", "post mini M-step"])
    def update(self, message, *args, **kwargs):
        if not self._gnuplot_path:
            return
        model = kwargs["model"]
        two_pop = hasattr(model, "split")
        can_plot_2 = two_pop and (model.split > model.model2.s[0])
        if two_pop:
            # plot split models
            x = np.cumsum(model.model1.s) * 2 * model.model1.N0
            y = model.model1.stepwise_values() * model.model1.N0
            z = model.model2.stepwise_values() * model.model2.N0
            data = "\n".join([",".join(map(str, row)) for row in zip(x, y)])
            if can_plot_2:
                data += "\n" * 3
                data += "\n".join(
                    [
                        ",".join(map(str, row))
                        for row in zip(x, z)
                        if row[0] <= 2 * model.model1.N0 * model.split
                    ]
                )
        else:
            x = np.cumsum(model.s) * 2 * model.N0
            y = model.stepwise_values() * model.N0
            u = model._knots * 2 * model.N0
            v = np.exp(model[:].astype("float")) * model.N0
            data = "\n".join([",".join(map(str, row)) for row in zip(x, y)])
            data += "\n" * 3
            data += "\n".join([",".join(map(str, row)) for row in zip(u, v)])

        graphs = ""
        for log_y in [True]:
            # Fire up the plot process and let'er rip.
            gnuplot = subprocess.Popen(
                [self._gnuplot_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE
            )

            def write(x):
                x += "\n"
                gnuplot.stdin.write(x.encode())

            columns, lines = np.maximum(shutil.get_terminal_size(), [80, 25])
            width = columns - 2
            height = lines * 4 // 5
            write("set term dumb {} {}".format(width, height))
            write('set datafile separator ","')
            write('set xlabel "Generations"')
            write('set ylabel "N_e"')
            xr = [
                model.distinguished_model.knots[i] * 2 * model.distinguished_model.N0
                for i in [0, -(len(smcpp.defaults.additional_knots) + 1)]
            ]
            write("set xrange [%f:%f]" % tuple(xr))
            if log_y:
                write("set logscale xy")
            else:
                write("set logscale x")
            with tempfile.NamedTemporaryFile("wt") as f:
                plot_cmd = "plot '%s' i 0 with lines title 'Pop. 1'" % f.name
                if two_pop and can_plot_2:
                    plot_cmd += ", '' i 1 with lines title 'Pop. 2';"
                elif not two_pop:
                    plot_cmd += ", '' i 1 with points notitle;"
                write(plot_cmd)
                open(f.name, "wt").write(data)
                write("unset key")
                write("exit")
                (stdout, stderr) = gnuplot.communicate()
                graphs += stdout.decode()
        logfun = logger.debug if message == "post mini M-step" else logger.info
        logfun("Plot of current model:\n%s", graphs)
