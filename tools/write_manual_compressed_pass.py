#!/usr/bin/env python3
"""Write manually compressed fixes for the remaining S1K 599 edge cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from compress_s1k import parse_metadata, sample_id, validate_compression


MANUAL: dict[int, dict[str, str]] = {
    521: {
        "compressed_question": "Parallelogram $ABCD$: extend $DA$ through $A$ to $P$. Line $PC$ meets $AB$ at $Q$ and diagonal $DB$ at $R$. Given $PQ=735$ and $QR=112$, find $RC$.",
        "compressed_reasoning": r"""1. Use affine coordinates $A=(0,0),B=(1,0),D=(0,1),C=(1,1),P=(0,-p)$ with $p>0$.
2. On $PC$, write points as $P+\lambda(C-P)$. Intersections give $Q$ at $\lambda_Q=p/(p+1)$ and $R$ at $\lambda_R=(p+1)/(p+2)$.
3. Hence $PQ:QR:RC=\frac{p}{p+1}:\frac{1}{(p+1)(p+2)}:\frac{1}{p+2}$.
4. From $PQ/QR=735/112=105/16$, $p(p+2)=105/16$, so $p+1=11/4$. Thus $RC/QR=p+1=11/4$.
Answer: \boxed{308}""",
        "compressed_answer": "\\boxed{308}",
    },
    120: {
        "compressed_question": "Rotons have dispersion $E(p)=\\Delta+b(p-p_0)^2$ with $b>0$. Compute the 3D density of states $\\nu(E)$ for volume $V$ and identify the matching multiple-choice option.",
        "compressed_reasoning": r"""1. The density of states is $\nu(E)=\frac{V}{2\pi^2\hbar^3}\sum p^2|dp/dE|$ over all momentum branches.
2. From $E=\Delta+b(p-p_0)^2$, $p=p_0\pm\sqrt{(E-\Delta)/b}$ and $|dp/dE|=(2\sqrt b\sqrt{E-\Delta})^{-1}$.
3. Summing both branches gives $(p_0+s)^2+(p_0-s)^2=2(p_0^2+s^2)$ with $s^2=(E-\Delta)/b$.
4. Therefore $\nu(E)=\frac{V}{2\pi^2\hbar^3}\frac{p_0^2+(E-\Delta)/b}{\sqrt{b(E-\Delta)}}$, which is option D.
Answer: \boxed{\frac{V}{2\pi^2\hbar^3}\frac{p_0^2+(E-\Delta)/b}{\sqrt{b(E-\Delta)}}}""",
        "compressed_answer": r"\boxed{\frac{V}{2\pi^2\hbar^3}\frac{p_0^2+(E-\Delta)/b}{\sqrt{b(E-\Delta)}}}",
    },
    135: {
        "compressed_question": "Let $d\\ge2$. Prove that for any origin-symmetric convex polytope $K\\subset\\mathbb R^d$ and $\\varepsilon\\in(0,1)$, there is a polytope $L\\subset K$ with at most $C(d)\\varepsilon^{1-d}$ vertices and $(1-\\varepsilon)K\\subseteq L$.",
        "compressed_reasoning": r"""1. Let $h_K(u)=\max_{x\in K}\langle u,x\rangle$ and $S=\{u:h_K(u)=1\}$. Take an $\varepsilon$-net $\mathcal N\subset S$ in the $h_K$ metric; volumetric bounds give $|\mathcal N|\le C(d)\varepsilon^{1-d}$.
2. For each $v\in\mathcal N$, choose $x_v\in K$ with $\langle v,x_v\rangle=1$, and set $L=\operatorname{conv}\{x_v:v\in\mathcal N\}$. Then $L\subseteq K$ and $L$ has the required number of vertices.
3. For $u\in S$, choose $v\in\mathcal N$ with $h_K(u-v)\le\varepsilon$. Since $x_v\in K$, $\langle u,x_v\rangle\ge\langle v,x_v\rangle-h_K(u-v)\ge1-\varepsilon$.
4. Thus $h_L(u)\ge(1-\varepsilon)h_K(u)$ for all $u$ by homogeneity, hence $(1-\varepsilon)K\subseteq L\subseteq K$.
Answer: \boxed{C(d)\varepsilon^{1-d}}""",
        "compressed_answer": r"\boxed{C(d)\varepsilon^{1-d}}",
    },
    142: {
        "compressed_question": "For $f:\\mathbb N_0\\to\\mathbb N_0$ with $f(0)=0$, $f(2n)=2f(n)$, $f(2n+1)=n+2f(n)$, find $L=\\{n:f(n)<f(n+1)\\}$, $E=\\{n:f(n)=f(n+1)\\}$, $G=\\{n:f(n)>f(n+1)\\}$, and $a_k=\\max_{0\\le n\\le2^k} f(n)$.",
        "compressed_reasoning": r"""1. Put $L_1=\{2k:k>0\}$, $E_1=\{0\}\cup\{4k+1:k\ge0\}$, $G_1=\{4k+3:k\ge0\}$; these partition $\mathbb N_0$.
2. $f(2k+1)-f(2k)=k>0$, so $L=L_1$. Also $f(4k+1)=2k+4f(k)=f(4k+2)$, so $E=E_1$.
3. Inductively $f(n+1)-f(n)\le n$. Then $f(4k+4)-f(4k+3)=4(f(k+1)-f(k))-(4k+1)<0$, so $G=G_1$.
4. The maximum on $[0,2^k]$ occurs at $2^k-1$, giving $a_k=2a_{k-1}+2^{k-1}-1$ with $a_0=0$. Solving gives $a_k=k2^{k-1}-2^k+1$.
Answer: \boxed{(a)\ L=\{2k:k>0\},\ E=\{0\}\cup\{4k+1:k\ge0\},\ G=\{4k+3:k\ge0\};\ (b)\ a_k=k2^{k-1}-2^k+1}""",
        "compressed_answer": "\\boxed{(a)\\ L=\\{2k:k>0\\},\\ E=\\{0\\}\\cup\\{4k+1:k\\ge0\\},\\ G=\\{4k+3:k\\ge0\\};\\ (b)\\ a_k=k2^{k-1}-2^k+1}",
    },
    159: {
        "compressed_question": "Speed density $\\rho(t,v)$ satisfies $\\rho_t+((u(t)-v)\\rho)_v=\\rho_{vv}$ with $u=u_0+u_1N$, $N=\\int_0^\\infty v\\rho dv$. (1) Prove $u_1>1$ can make $N$ unbounded. (2) For periodic $p_t+vp_x+((u-v)p)_v=p_{vv}$ on $x\\in[0,2\\pi]$, prove/disprove spatial equilibration.",
        "compressed_reasoning": r"""1. Let $M(t)=\int_{-\infty}^{\infty}v\rho(t,v)dv$. Integration by parts gives $M'=u(t)-M=u_0+u_1N-M$.
2. Since $M\le N$ and $N\ge0$, if $u_1>1$ then $M'\ge u_0+(u_1-1)N\ge u_0>0$; hence $M$ and therefore $N$ can grow unbounded.
3. Expand $p=(2\pi)^{-1}\sum_k p_k(t,v)e^{ikx}$. For $k\ne0$, Fourier transform in $v$ gives $\partial_t\hat p_k+(\xi-k)\partial_\xi\hat p_k=-(i\xi u(t)+\xi^2)\hat p_k$.
4. Characteristics yield an inverse-transform formula with damping $\|p_k(t,\cdot)\|_1\le\|p_{0,k}\|_1e^{-k^2D(t)}$, where $D(t)=t-2(1-e^{-t})/(1+e^{-t})>0$ and grows like $t$. Thus all $k\ne0$ modes decay and only the spatial average remains.
Answer: \boxed{\text{(1) }u_1>1\text{ can make }N(t)\text{ unbounded; (2) }p\text{ becomes spatially uniform.}}""",
        "compressed_answer": "\\boxed{\\text{(1) }u_1>1\\text{ can make }N(t)\\text{ unbounded; (2) }p\\text{ becomes spatially uniform.}}",
    },
    162: {
        "compressed_question": "Let $K$ be a closed subset of the closed unit ball in $\\mathbb R^3$. Suppose chords of $S^2$ disjoint from $K$ are dense in the sense that every pair of endpoints can be approximated by endpoints of such a chord. Prove there is dense $H\\subset S^2$ such that every chord joining two points of $H$ is disjoint from $K$.",
        "compressed_reasoning": r"""1. Let $R=\{(X,Y)\in S^2\times S^2:XY\cap K=\emptyset\}$. Since $K$ is compact and chords vary continuously, $R$ is open. The hypothesis makes $R$ dense.
2. For any fixed $X$, the section $R_X=\{Y:(X,Y)\in R\}$ is dense: approximate $(X,Y)$ by a good chord and use openness to move the first endpoint back to $X$.
3. Choose a countable base $(U_n)$ of $S^2$. Inductively pick $h_n\in U_n\cap\bigcap_{i<n}R_{h_i}$, possible because a finite intersection of dense open sets is dense.
4. Then $H=\{h_n:n\ge1\}$ is dense, and for $i<j$ the construction gives $(h_i,h_j)\in R$, so every chord joining two points of $H$ misses $K$.
Answer: \boxed{H\text{ exists as required}}""",
        "compressed_answer": "\\boxed{H\\text{ exists as required}}",
    },
    171: {
        "compressed_question": "Ellipse $x^2/4+y^2/3=1$. $H=(\\alpha,0)$; the vertical through $H$ meets the ellipse at $E$ and auxiliary circle $x^2+y^2=4$ at $F$. Tangent at $E$ meets $+x$-axis at $G$, and $OF$ makes angle $\\phi$. Match $\\phi=\\pi/4,\\pi/3,\\pi/6,\\pi/12$ to areas P,Q,R,S,T; options A:R,S,Q,P; B:R,T,S,P; C:Q,T,S,P; D:Q,S,Q,P.",
        "compressed_reasoning": r"""1. $E=(\alpha,\sqrt{3-3\alpha^2/4})$ and $F=(\alpha,\sqrt{4-\alpha^2})$.
2. Tangent at $E$ is $x\alpha/4+y\sqrt{3-3\alpha^2/4}/3=1$, so $G=(4/\alpha,0)$.
3. Area $[FGH]=\frac12(4/\alpha-\alpha)\sqrt{4-\alpha^2}=\frac{(4-\alpha^2)^{3/2}}{2\alpha}$.
4. Since $\tan\phi=\sqrt{4-\alpha^2}/\alpha$, $\alpha=2\cos\phi$, hence area $=2\sin^3\phi/\cos\phi$.
5. Values for $\pi/4,\pi/3,\pi/6,\pi/12$ are $1,3\sqrt3/2,1/(2\sqrt3),(\sqrt3-1)^4/8$, i.e. Q,T,S,P.
Answer: \boxed{C}""",
        "compressed_answer": "\\boxed{C}",
    },
    218: {
        "compressed_question": "For twice differentiable $f(x,y)$ on $x,y\\ge1$ satisfying $xf_x+yf_y=xy\\ln(xy)$ and $x^2f_{xx}+y^2f_{yy}=xy$, define $T(f,s)=f(s+1,s+1)-f(s+1,s)-f(s,s+1)+f(s,s)$ and $m(f)=\\min_{s\\ge1}T(f,s)$. Determine $m(f)$ and show it is independent of $f$.",
        "compressed_reasoning": r"""1. Differentiate $xf_x+yf_y=xy\ln(xy)$ in $x$ and $y$, multiply by $x,y$, add, and use the two given equations to obtain $f_{xy}=\frac12(\ln(xy)+1)$.
2. Integrating gives $f=\frac12xy\ln(xy)-\frac12xy+D(x)+E(y)$; substituting in the first PDE gives $D(x)=A\ln x$, $E(y)=-A\ln y$, plus a constant.
3. The mixed difference $T$ kills $A\ln(x/y)$ and constants, so $m(f)$ is independent of $f$. Direct substitution in the remaining term gives $T(f,s)=\ln\frac{(s+1)^{s+1}}{s^s}+\frac12$.
4. This expression is increasing for $s\ge1$, so the minimum occurs at $s=1$ and equals $\log4+\frac12$.
Answer: \boxed{\frac{1}{2}+\log 4}""",
        "compressed_answer": r"\boxed{\frac{1}{2}+\log 4}",
    },
    250: {
        "compressed_question": "Let $x,y,z>0$ and $(x/y+y/z+z/x)+(y/x+z/y+x/z)=8$. Find the minimum of $(x/y+y/z+z/x)(y/x+z/y+x/z)$.",
        "compressed_reasoning": r"""1. Let $P=(x/y+y/z+z/x)(y/x+z/y+x/z)$ and $S=8$. Expanding $2P=S^2-A^2-B^2$ gives $2P=51-U$, where $U=(x^2+y^2+z^2)(1/x^2+1/y^2+1/z^2)$.
2. Also $(x+y+z)(1/x+1/y+1/z)=11$ and $(xy+xz+yz)(1/(xy)+1/(xz)+1/(yz))=11$.
3. Cauchy-Schwarz gives $11\ge\sqrt U+2\sqrt{11}$, hence $U\le(11-2\sqrt{11})^2=165-44\sqrt{11}$.
4. Thus $2P\ge51-(165-44\sqrt{11})=44\sqrt{11}-114$, so $P\ge22\sqrt{11}-57$. Equality is attainable by the equality case described by $a^2/b=\sqrt{11}$.
Answer: \boxed{22\sqrt{11}-57}""",
        "compressed_answer": r"\boxed{22\sqrt{11}-57}",
    },
    341: {
        "compressed_question": "Count integer sequences $a_1,\\dots,a_5$ with each $a_i\\le1$ and all partial sums nonnegative.",
        "compressed_reasoning": r"""1. Let $dp[i][k]$ count length-$i$ sequences with partial sum $k\ge0$. Since $a_i=k-k'\le1$, $dp[i][k]=\sum_{k'\ge k-1}dp[i-1][k']$.
2. Iterating for five steps gives final counts by ending sum $k=0,1,2,3,4,5$: $42,42,28,14,5,1$.
3. Summing all allowed final sums gives $42+42+28+14+5+1=132$.
Answer: \boxed{132}""",
        "compressed_answer": "\\boxed{132}",
    },
    429: {
        "compressed_question": "For vectors $a,b,c\\in\\mathbb R^n$, prove $(\\|a\\|\\langle b,c\\rangle)^2+(\\|b\\|\\langle a,c\\rangle)^2\\le\\|a\\|\\|b\\|(\\|a\\|\\|b\\|+|\\langle a,b\\rangle|)\\|c\\|^2$.",
        "compressed_reasoning": r"""1. If one vector is zero the claim is immediate. Otherwise set $u=a/\|a\|$, $v=b/\|b\|$, $w=c/\|c\|$ and divide by $\|a\|^2\|b\|^2\|c\|^2$. It remains to show $\langle u,w\rangle^2+\langle v,w\rangle^2\le1+|\langle u,v\rangle|$.
2. Let $r=\langle u,v\rangle$, $s=\langle v,w\rangle$, $t=\langle w,u\rangle$. The Gram determinant is nonnegative: $1-r^2-s^2-t^2+2rst\ge0$.
3. Hence $s^2+t^2\le1-r^2+2|r||st|\le1-r^2+|r|(s^2+t^2)$. If $|r|<1$, divide by $1-|r|$ to get $s^2+t^2\le1+|r|$; if $|r|=1$ it is trivial.
4. Multiplying back gives the required inequality.
Answer: \boxed{(\|a\|\langle b,c\rangle)^2+(\|b\|\langle a,c\rangle)^2\le\|a\|\|b\|(\|a\|\|b\|+|\langle a,b\rangle|)\|c\|^2}""",
        "compressed_answer": "\\boxed{(\\|a\\|\\langle b,c\\rangle)^2+(\\|b\\|\\langle a,c\\rangle)^2\\le\\|a\\|\\|b\\|(\\|a\\|\\|b\\|+|\\langle a,b\\rangle|)\\|c\\|^2}",
    },
    439: {
        "compressed_question": "For Brownian motion $W(0)=0$, let $X(s,t)=\\inf_{u\\in[s,t]}W(u)$. For $t>1,\\varepsilon>0$, find the density $f_{t,\\varepsilon}(x)$ of $W(1)=x\\ge0$ conditioned on $X(0,t)>-\\varepsilon$. Also find the density $g$ of $R=(G_1^2+G_2^2+G_3^2)^{1/2}$ and show it is the $t\\to\\infty,\\varepsilon\\downarrow0$ limit.",
        "compressed_reasoning": r"""1. Let $\varphi(x)=(2\pi)^{-1/2}e^{-x^2/2}$. Reflection gives $P(X(0,t)>-\varepsilon)=2\Phi(\varepsilon/\sqrt t)-1$.
2. Conditioning on $W_1=x$ and using the Markov property, the survival after time $1$ contributes $2\Phi((x+\varepsilon)/\sqrt{t-1})-1$.
3. Reflection on $[0,1]$ gives the first-step density factor $\varphi(x)-\varphi(x+2\varepsilon)=\varphi(x)(1-e^{-2\varepsilon x-2\varepsilon^2})$. Dividing by the denominator yields $f_{t,\varepsilon}$.
4. Since $R^2\sim\chi^2_3$, $g(x)=\sqrt{2/\pi}x^2e^{-x^2/2}$ for $x>0$. L'Hospital as $\varepsilon\downarrow0$ and then $t\to\infty$ gives the same expression.
Answer: \boxed{f_{t,\varepsilon}(x)=\frac{\varphi(x)(1-e^{-2\varepsilon x-2\varepsilon^2})(2\Phi((x+\varepsilon)/\sqrt{t-1})-1)}{2\Phi(\varepsilon/\sqrt t)-1},\quad g(x)=\sqrt{2/\pi}x^2e^{-x^2/2}}""",
        "compressed_answer": r"\boxed{f_{t,\varepsilon}(x)=\frac{\varphi(x)(1-e^{-2\varepsilon x-2\varepsilon^2})(2\Phi((x+\varepsilon)/\sqrt{t-1})-1)}{2\Phi(\varepsilon/\sqrt t)-1},\quad g(x)=\sqrt{2/\pi}x^2e^{-x^2/2}}",
    },
    564: {
        "compressed_question": "In triangle $ABC$, $D,E,F$ are midpoints of $BC,AC,AB$, and $P,Q,R$ are midpoints of $AD,BE,CF$. Compute $(AQ^2+AR^2+BP^2+BR^2+CP^2+CQ^2)/(AB^2+AC^2+BC^2)$.",
        "compressed_reasoning": r"""1. Write position vectors as $a,b,c$. Then $p=(2a+b+c)/4$, $q=(a+2b+c)/4$, $r=(a+b+2c)/4$.
2. Expanding the six squared distances and collecting symmetric terms gives numerator $\frac{28}{16}(a\cdot a+b\cdot b+c\cdot c-a\cdot b-a\cdot c-b\cdot c)$.
3. Also $AB^2+AC^2+BC^2=2(a\cdot a+b\cdot b+c\cdot c-a\cdot b-a\cdot c-b\cdot c)$.
4. The ratio is therefore $(28/16)/2=7/8$.
Answer: \boxed{\frac{7}{8}}""",
        "compressed_answer": r"\boxed{\frac{7}{8}}",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/s1K_train_599.json")
    parser.add_argument("--output_jsonl", default="data/compression_pilot/deepseek_q220_r512_parts_v3/manual_pass_1/output.jsonl")
    parser.add_argument("--max_question_tokens", type=int, default=220)
    parser.add_argument("--max_reasoning_tokens", type=int, default=512)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    samples = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    all_ok = True
    for index in sorted(MANUAL):
        item = samples[index]
        compressed = MANUAL[index]
        validation = validate_compression(item, compressed, args.max_question_tokens, args.max_reasoning_tokens)
        print(index, validation)
        all_ok = all_ok and bool(validation.get("ok"))
        rows.append(
            {
                "index": index,
                "sample_id": sample_id(index, item),
                "metadata": parse_metadata(item.get("metadata")),
                "source_type": item.get("source_type"),
                "original": {
                    "question": item.get("question"),
                    "solution": item.get("solution"),
                    "deepseek_attempt": item.get("deepseek_attempt"),
                },
                "compressed": compressed,
                "validation": validation,
                "_manual_note": "manual_pass_1",
            }
        )
    print("ALL_OK", all_ok)
    if args.write:
        out = Path(args.output_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        print(f"WROTE {out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
