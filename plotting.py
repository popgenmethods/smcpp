themes = """Palette	Palette Column	Palette Row	PC (c,r)	RGB	One
Tableau 10	1	2	T 10: 1,2	255.127.14	1
Tableau 10	1	4	T 10: 1,4	214.39.40	1
Tableau 10	1	5	T 10: 1,5	148.103.189	1
Tableau 10	1	3	T 10: 1,3	44.160.44	1
Tableau 10	1	1	T 10: 1,1	31.119.180	1
Tableau 10	2	2	T 10: 2,2	227.119.194	1
Tableau 10	2	4	T 10: 2,4	188.189.34	1
Tableau 10	2	1	T 10: 2,1	140.86.75	1
Tableau 10	2	3	T 10: 2,3	127.127.127	1
Tableau 10	2	5	T 10: 2,5	23.190.207	1"""

palette = [(.2, .2, .2)]
for row in themes.split("\n")[1:]:
    fields = row.strip().split("\t")
    rgb = tuple([int(x) / 255. for x in fields[-2].split(".")])
    palette.append(rgb)

def pretty_plot():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()  
    return fig, ax

def save_pdf(plt, filename):
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages(filename)
    plt.savefig(pp, format='pdf')
    pp.close()
