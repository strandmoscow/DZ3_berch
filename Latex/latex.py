from jinja2 import Environment, FileSystemLoader
from math import factorial


def matr_to_table_r(mat, n, delim):
    s = ""
    for i in mat:
        for x in i:
           s += f'{round(x, n)}'
           s += f'{delim}'
        s = s[:-len(delim)]
        s = s + " \\\\\n"
    s = s[:-3]
    return s


def matr_to_table_r_int(mat, delim):
    s = ""
    for i in mat:
        for x in i:
           s += f'{int(x)}'
           s += f'{delim}'
        s = s[:-len(delim)]
        s = s + " \\\\\n"
    s = s[:-3]
    return s


def string_for_renumerating_states(nodes):
    s = ""
    j = 0
    for i in nodes:
        name = i.get_name()
        if name == "stA":
            s = s + "$S_{" + str(j) + "} = S^{term}_B$, "
        elif name == "stB":
            s = s + "$S_{" + str(j) + "} = S_{term}^A$, "
        else:
            s = s + "$S_{" + str(j) + "} = S^{" + name[1] + name[2] + "}_{" +  name[3] + name[4] + "}$, "
        j = j + 1
    s = s[:-2]
    return s


def P_funcs(paths, lambda_A, lambda_B, Na, Nb):
    s = ""
    for path in paths:
        if paths[path][0] > 0:
            k = (Na * lambda_A) ** paths[path][1] * (Nb * lambda_B) ** paths[path][2] / factorial(paths[path][1]) / factorial(paths[path][2])
            s = s + "P_{" + str(paths[path][0]) + "} (t)=" + str(k)[:-2] + "e^{-" + str(Na * lambda_A + Nb * lambda_B) + "t} t^" + str(paths[path][1] + paths[path][2]) + " \\\\ \n"
    s = s[:-1]
    return s


def P_term(paths, lambda_A, lambda_B, Na, Nb):
    s = ""
    l = [0] * 10
    for path in paths:
        if paths[path][0] > 0:
            k = (Na * lambda_A) ** paths[path][1] * (Nb * lambda_B) ** paths[path][2] / factorial(paths[path][1]) / factorial(paths[path][2])
            l[paths[path][1] + paths[path][2]] = l[paths[path][1] + paths[path][2]] + k
    i = 9

    while i >= 0:
        if l[i] > 0:
            s = s + str(round(l[i], 4))[:-2] + "t^" + str(i) + " + "
        i = i - 1
    s = s + "1"
    return s


def make_latex(var, g, name, name_short, lambda_A, lambda_B, Na, Nb, Ra, Rb, graph_tex, nodes, matrix_np, kolmogorov, paths, mu, st_o, sred_o):
    # Jinja init
    environment = Environment(
        loader=FileSystemLoader("Latex/templates/")
    )

    # Preamble text
    base_template = environment.get_template("educmm_lab_Variant_N_M-id.tex")
    base_res_file_name = "Latex/res/labs/educmm_txb_COMPMATHLAB-Solution_N_M/educmm_lab_Variant_N_M-id.tex"
    base_text = base_template.render(
        author_name="{" + str(name) + "}",
        author_name_short="{" + str(name_short) + "}",
        group="{" + f"РК6-8{g}б" + "}",
        variant="{" + str(var) + "}"
    )

    with open(base_res_file_name, mode="w+", encoding="utf-8") as base:
        base.write(base_text)
        print(f"... wrote {base_res_file_name}")

    # Main text
    latex_text_template = environment.get_template("educmm_txb_COMPMATHLAB-Solution_N_M.tex")
    latex_text_file_name = f"Latex/res/labs/educmm_txb_COMPMATHLAB-Solution_N_M.tex"
    latex_text = latex_text_template.render(
        g=g,
        var=var,
        lambda_A=lambda_A,
        lambda_B=lambda_B,
        Na=Na,
        Nb=Nb,
        Ra=Ra,
        Rb=Rb,
        G=graph_tex,
        renumerate=string_for_renumerating_states(nodes),
        mat=matr_to_table_r_int(matrix_np, ' & '),
        kolmogorov=kolmogorov,
        P_funcs=P_funcs(paths, lambda_A, lambda_B, Na, Nb),
        pathsN=len(paths),
        P_term=P_term(paths, lambda_A, lambda_B, Na, Nb),
        mu=mu,
        st_o=st_o,
        sred_o=sred_o
    )
    with open(latex_text_file_name, mode="w+", encoding="utf-8") as text:
        text.write(latex_text)
        print(f"... wrote {latex_text_file_name}")



