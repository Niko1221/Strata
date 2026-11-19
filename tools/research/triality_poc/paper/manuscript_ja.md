---
title: "Triality条件付き直交Query変換と量子化誤差制御によるLLM全体への寄与"
subtitle: "全ネットワーク摂動解析・関連研究・PyTorch CPU/CUDA対応PoC"
author: "ユーザー提供のStrata・Bonsai・TurboQuant統合仕様に基づく研究草稿"
date: "2026年10月23日・未査読の技術報告 v1"
lang: ja-JP
fontsize: 10pt
geometry: margin=23mm
papersize: a4
mainfont: DejaVu Serif
sansfont: DejaVu Sans
monofont: DejaVu Sans Mono
CJKmainfont: Noto Serif CJK JP
colorlinks: true
linkcolor: black
urlcolor: blue
---

# 要旨

本稿は、Spin(8)のnormed Trialityに由来する入力依存の直交Query変換と、多表現KV量子化・共有残差補正を、言語モデル全体の出力忠実度へ結び付けるための理論枠組みと研究用PoCを示す。既存の回転量子化が主に全精度時の関数保存を利用するのに対し、提案するQuery側アダプターは学習後のAttention関数を変更する。したがって、幾何学的正当性、量子化補償、課題能力の向上、実行速度は別々に検証する。

主な解析内容は、(i) 7つのClifford生成元によるQuery接空間の完全性とブロック共有の制約、(ii) softmaxのFisher情報に基づく誤差計量と補償可能部分空間、(iii) 共有QJLのbias–variance trade-off、(iv) 局所Attention摂動から残差層・再帰状態・最終語彙logit・自己回帰分布への誤差伝播である。既知の恒等式・理論からの帰結と、本設計への適用上の仮説を区別する。

PoCは純粋なPyTorchで実装し、CPU/CUDAを明示選択する。今回の実行環境はPyTorch 2.10.0+cpuであり、CUDA実行は未検証である。CPUでは25件の単体試験、12件の数値チェック、7件の限定的なソース変異検出を完了した。独立に学習した3個の小規模自己回帰Transformerによる合成課題では、三表現2bit経路にQアダプターを追加すると、教師との語彙KLの平均が0.027913から0.023622 nat/tokenへ低下した。しかし対応差の探索的95%信頼区間はゼロを含み、単一6bit経路はさらに小さい保存量で大幅に高精度だった。したがって本結果は、新方式の普遍的優位性や27Bモデルの改善を示さない。Bonsai 27B、Strata大型Q2_0、TheTom形式での実機評価は今後の必須検証である。

**キーワード:** Spin(8)、Triality、直交Queryアダプター、KV量子化、Fisher情報、誤差伝播、蒸留、再現可能性。

**研究上の位置付け:** 数理整理と検証可能な仮説を含む技術草稿であり、査読済み論文でも、実用LLMへの統合完了報告でもない。設計原案はユーザー提供仕様、文献調査・数理整理・PoC作成支援はChatGPTによる。著者・所属・貢献表記は、投稿前に人間の研究責任者が確定する。

# 1. 問題設定と主張の範囲

TransformerのAttentionはQueryとKeyの関係から重みを決め、Valueを集約する[1]。低bit化では、この関係と取り出される値の両方に誤差が入る。重要なのは、KVベクトルの再構成誤差を小さくするだけでなく、その誤差が最終の次トークン分布にどれだけ影響するかを評価することである。

本稿では、以下を別々の操作として扱う。

1. **関数を変える適応:** 入力から生成した直交行列をQだけに適用する。学習後の関数は元モデルと一般に異なる。
2. **関数を保つ基底変更:** QとKに対応する同じ直交基底変更を行い、その間の量子化誤差を制御する。量子化を外せば内積を保存する。
3. **数値・システム実装:** 量子化状態のpacking、キャッシュ操作、online softmax、融合カーネル、実メモリと遅延の測定。

この区別により、「代数的に正しい」から「LLMの課題性能が向上する」への飛躍を避ける。本稿で主張するのは、全ネットワークへの寄与を解析・測定できる枠組みと、自己完結した参照PoCである。普遍的な高速化、新しい知識の獲得、一般的な推論能力の向上、全層のSpin(8)同変性は主張しない。

## 1.1 対象となる実モデルと、今回の実験対象

製品化時の主対象は、ユーザー提供情報による `zapabobouj/Hikari-Bonsai-27B-Nimble-LoRA-GGUF` である。期待SHA-256は次の値とする。

```
0ab2a1f237b73e5002ecd8c9b764249b90ee59eca749a6226c44bb43b1036a13
```

提示された構成はqwen35、64層、Qヘッド24、KVヘッド4、幅256、通常Attention16層・再帰48層である。このローカルファイルを本稿作成環境で再取得・再ハッシュしたわけではない。元のNimble学習済み重みは読み取り専用とし、追加学習はsidecarに保存する。Strata大型Q2_0とTheTom形式の実モデルも完成判定から除外しない。

今回実際に動かしたのは、別に実装した小規模な合成言語モデルである。以下の数値を、27B・125B・Windows・RTX 5060 Tiの測定値として流用してはならない。

# 2. 関連研究と本研究の境界

## 2.1 群・幾何学とAttention

Baezによる八元数・Spin表現・Trialityの解説[2]は本構成の数学的基礎である。LieTransformer[3]とGeometric Algebra Transformer[4]は、対象データの群作用に対する同変性を構成する研究である。本方式は内部特徴の8次元ブロックに群論的制約を導入するが、自然言語がこの群対称性を持つという仮定も、全モデルの同変性証明も置かない。Clifford代数を用いることと、同変なTransformerを構築することは同義ではない。

QK-Norm[5]はQueryとKeyの正規化によりAttentionのスケールを扱い、RoFormer[6]は位置依存の回転を導入する。nGPT[7]はさらに広い範囲の表現を球面上で扱う。本方式は既存Q正規化の後に方向の適応を加える点で関連するが、Qのノルムを保存するだけで、nGPTの全体構造や学習効率を再現するものではない。

## 2.2 パラメーター効率的な適応と蒸留

LoRA[8]は低rankの重み更新を利用する。OFT/COFT[9]は直交制約と変形量の制約を利用し、BOFT[10]はそのパラメーター効率を改善する。本稿のrank 8は、7次元制御量を生成するネットワークの中間幅であり、線形重み差分のrankを直接意味しない。また、入力依存のQ変換は通常のLoRA形式にそのままマージできない。

教師分布への忠実度を用いる方針は知識蒸留[11]に関連する。ただし、教師との一致を改善することと、独立した課題の正答率を向上させることは別である。教師自身の誤りを忠実に再現する可能性も残る。

## 2.3 回転量子化・KV圧縮

QuaRot[12]とSpinQuant[13]は、全精度時の関数保存を利用する回転と低bit化の組合せを研究している。QuIP#[14]はHadamardによるincoherence処理と格子codebookを扱い、GPTQ[15]は層の誤差感度を考慮した重み量子化を扱う。これらは量子化しやすい座標を選ぶ動機を与えるが、SpinQuantという名称は、本稿の厳密Spin(8)半スピノル実装を意味しない。またQuIP#の$E_8$格子と、本稿の三表現は同一の構成ではない。

KIVI[16]はKとVの分布・量子化単位の非対称性に着目する。QJL[17]とTurboQuant[18]は、ランダム射影、量子化、内積推定の誤差を扱う。本稿の共有残差QJLはこれらの考え方に基づくが、既存論文のメモリ・品質保証を、任意の実装・固定乱数・学習済みアダプターへ無条件に移さない。

## 2.4 安定性・情報幾何・システム

通常のdot-product self-attentionには、入力領域を無制限にすると大域Lipschitz性が成り立たない場合がある[19]。系列長・正規化・Attention分布を考慮した局所解析[20,25]は、各Q変換が直交であることだけでは全モデルの安定性を保証しない理由になる。Fisher情報・自然勾配の理論[21]は、確率分布の変化に即した局所計量を与える。

Gated DeltaNet[22]のような再帰層を含むモデルでは、変更していない再帰演算も上流入力の変化を受ける。FlashAttention[23]はIOを削減する正確なAttention実装、GQA[24]はKV共有の構造的削減であり、本方式とは別の改良軸である。最近のKV量子化解析[26]にも、スコアの偏り・分散とsoftmax後の品質が一致しない実験がある。その特定条件での結果を、QJL全般の不可能性定理とは解釈しない。

## 2.5 新規性として検討すべきもの

本稿の新規性候補は、normed Trialityから得る入力条件付きQ変換、三表現の共有残差補正、Attention/語彙分布のFisher計量、全ネットワーク誤差伝播を同じ検証体系で扱う点である。ただし、各恒等式、直交適応、回転量子化、蒸留、Fisher計量はそれぞれ先行研究を持つ。「世界初」「既存研究にない」とは記さない。今回の調査は関連文献の選択的レビューであり、体系的レビューや特許調査ではない。

# 3. モデルと演算の定義

## 3.1 正規化されたTriality

正のClifford規約

$$
\Gamma_i\Gamma_j+\Gamma_j\Gamma_i=2\delta_{ij}I_{16}
\tag{1}
$$

を固定する。同じ順序付き平面回転列からSpin元を構成し、ベクトル表現$R_v$と左右半スピノル表現$R_+,R_-$を得る。独立な3個の直交行列を「厳密Triality」と呼ばない。Baez[2]の符号規約との対応は明示し、2πでのスピノル符号反転と4πでの復帰を試験する。

正規化された橋渡しテンソルを$L_i$とし、$L(v)=\sum_i v_iL_i$と置く。

$$
T(v,p,c)=p^\top L(v)c,
\qquad
v^\top C(c)p=T(v,p,c).
\tag{2}
$$

ここでは、単なる不変三重線形形式ではなく、$\|L(v)c\|=\|v\|\|c\|$を満たすnormed Trialityを使う。

## 3.2 入力依存Queryアダプター

単位$c_0$と入力依存の単位スピノル$c(q)$に対し、

$$
A(c)=C(c)C(c_0)^\top,
\qquad q'_b=A(c(q))q_b
\tag{3}
$$

をヘッド内の8次元ブロック$b$へ適用する。端数は恒等変換とする。同じヘッド内の全ブロックが7個の制御量を共有する。Q正規化後・RoPE/MRoPE前への挿入を基本とし、元の出力gate、mask、scaleは保存する。

$c_0=e_0$、$E=[e_1,\ldots,e_7]$、$\tau=\tan(\pi/8)$として、滑らかなチャートを

$$
u(a)=\frac{\tau a}{\sqrt{\tau^2+\|a\|^2}},
\qquad
c(a)=\frac{c_0+E u(a)}{\sqrt{1+\|u(a)\|^2}}
\tag{4}
$$

とする。制御器は$a=W_{\rm out}\tanh(W_{\rm in}q+b_{\rm in})+b_{\rm out}$で、$W_{\rm in}$はrank 8の中間幅を持つ。$W_{\rm out},b_{\rm out}$のみゼロ初期化する。

**役割の区別:** 非量子化でも$A(c)\neq I$ならAttentionの関数は変わる。元モデルとの一致条件は、アダプター無効時・恒等初期化時、または同じアダプターを固定したcodec有無の比較に限る。

## 3.3 三表現codecと共有QJL

位置処理後の座標に対し、$U_r=B_rF$を用いる。$B_r$は同一Spin元の表現から作ったブロック対角行列、$F$は順序を固定した正規化Hadamardと符号反転である。

$$
\hat k_r=U_r^\top\mathcal Q_r(U_rk),\qquad
\bar k=\sum_r\alpha_r\hat k_r,\qquad e=k-\bar k,
\tag{5}
$$

$$
\alpha_r\ge0,\quad\sum_r\alpha_r=1,\quad
\alpha_r^{(0)}=1/3.
$$

$\mathcal Q_r$は量子化と復元を合わせた演算である。QJL有効時は、独立Gaussian行を持つ$G\in\mathbb R^{m\times d}$を使って、raw scoreを

$$
\hat s_{\rm raw}(q,k)=q^\top\bar k+
\beta\sqrt{\frac\pi2}\frac{\|e\|}{m}
\operatorname{sign}(Ge)^\top Gq
\tag{6}
$$

とする。$\beta=0$を既定とし、不要な残差cacheは確保しない。scale、bias、maskを適用した後、softmaxと共有V積をそれぞれ1回だけ行う。三表現の線形和は第4の集約結果であり、第4の独立表現ではない。

# 4. 局所幾何から得られる性質

## 命題1：直交性・恒等初期化・条件数

式(2)より$C(c)^\top v=L(v)c$であるから、

$$
C(c)C(c)^\top=\|c\|^2I.
$$

単位$c,c_0$に対し、$A^\top A=I$、$A(c_0)=I$が成立する。単位球面$S^7$は連結で、$\det C(c)$は連続な$\pm1$なので符号一定であり、$\det A=+1$となる。よって固定した線形演算$A$の2-ノルム条件数は1である。

これはノルム保存の証明であり、意味保存やタスク品質保存の証明ではない。$C(c)$単体のdetを+1と仮定する必要もない。

## 命題2：接空間の完全性と、ヘッド共有による制限

$J_a=C(e_a)C(c_0)^\top$とすると、

$$
J_a^\top=-J_a,\qquad
J_aJ_b+J_bJ_a=-2\delta_{ab}I.
\tag{7}
$$

したがって、非ゼロ$q\in\mathbb R^8$について

$$
q^\top J_aq=0,\qquad
(J_aq)^\top(J_bq)=\delta_{ab}\|q\|^2.
\tag{8}
$$

$J_aq$の7方向は球面の接空間を張る。単一8次元Queryを局所的に任意の接方向へ動かすには十分である。ただし、幅256の32ブロックで制御量を共有すると、可動方向は依然として7方向であり、256次元球面の255方向全体を独立に操作できるわけではない。

$J_a$の7個と$J_aJ_b$の21個は反対称行列の28次元空間を張る。したがって、これらが生成するLie代数は$\mathfrak{so}(8)$である。しかし単一ステップの7パラメーター族は積について閉じない。任意個の小回転を合成できる理想的制御系の到達性と、有限層・共有制御器・非線形演算を持つ実LLMの表現能力は区別する。PoCは行列rank 28を検査するが、LLMの普遍近似性を示すものではない。

## 命題3：方向変位の上限と非線形ヤコビアン

$c=\cos\theta c_0+\sin\theta E u$、$\|u\|=1$なら、

$$
A(c)=\cos\theta I+\sin\theta J(u)=\exp(\theta J(u)),
\quad J(u)^2=-I.
\tag{9}
$$

よって$\|A-I\|_2=2\sin(|\theta|/2)$であり、$|\theta|\le\pi/8$なら

$$
\|q'-q\|\le 0.390181\,\|q\|.
\tag{10}
$$

一方、$f(q)=A(c(q))q$の微分は

$$
Df(q)[h]=A(c(q))h+DA(c(q))[h]q.
\tag{11}
$$

直交性だけで$\|Df\|=1$、$\det Df=1$、勾配爆発なしとは言えない。正規化チャートの微分ノルムが1以下であり、tanhの微分も1以下であることから、単一ヘッドの有界領域では

$$
\|Df(q)\|_2\le1+\|q\|\,\|W_{\rm out}\|_2\|W_{\rm in}\|_2
\tag{12}
$$

という十分な上界が得られる。したがって角度制限に加え、制御器のスペクトルノルムや実ヤコビアンを測る必要がある[19,20,25]。量子化、丸め、分岐の不連続性までこの微分式で扱ったことにはならない。

# 5. Attentionの情報幾何と誤差補償

## 命題4：softmax KLの厳密式と局所計量

有限な同じmask support上で、$p=\operatorname{softmax}(s)$、$\hat p=\operatorname{softmax}(s+\varepsilon)$とする。

$$
D_{\rm KL}(p\|\hat p)
=\log\sum_jp_je^{\varepsilon_j}-\sum_jp_j\varepsilon_j.
\tag{13}
$$

小さな摂動では、

$$
D_{\rm KL}(p\|\hat p)
=\frac12\varepsilon^\top H_p\varepsilon+O(\|\varepsilon\|^3),
\quad H_p=\operatorname{diag}(p)-pp^\top.
\tag{14}
$$

$H_p$はcategorical分布のlogit座標におけるFisher行列である[21]。定数方向$\mathbf1$は零空間に入るため、全スコアへの共通加算は無害である。「内積のMSE」と「Attention分布のKL」は一致しない。以後のFisher最適化は、固定教師分布に対する局所近似であり、大誤差・mask変更・学習による分布変化に対する厳密保証ではない。

## 命題5：小さなQ制御器が修正できる部分空間

$\hat k_j=k_j+e_j$とし、恒等近傍で$\delta q=\sum_a\eta_a\mathcal J_aq$とする。位置回転を省略した記号では、

$$
\delta s_j\simeq b_j+F_{ja}\eta_a,
\quad b_j=\frac{q^\top e_j}{\sqrt d},
\quad F_{ja}=\frac{(\mathcal J_aq)^\top k_j}{\sqrt d}.
\tag{15}
$$

実際のRoPE前アダプターでは、$(\mathcal J_aq)^\top k_j$を$(P_t\mathcal J_aq)^\top(P_jk_j)$に置き換える。基底と位置処理を可換と仮定しない。

局所二次目的と正則化から、

$$
\eta_*=-\left(F^\top H_pF+\lambda I\right)^{-1}F^\top H_pb
\tag{16}
$$

が得られる。$\lambda>0$なら方程式は安定に解ける。$\lambda=0$の場合は擬似逆と最小ノルム解を使う。補償できるのは$F$の像に沿う誤差で、任意個のKeyに対する全誤差を7自由度で消せるわけではない。

式(16)はoracle診断である。元Kと量子化誤差をすべて使って各Queryで解くことを実用推論手順とはしない。学習した制御器がこの修正をどこまで近似できるかを検証するために用いる。

## 命題6：Valueまで含む歪み

$o=\sum_jp_jv_j$の一次変分は、

$$
\delta o\simeq\sum_jp_j(v_j-o)\delta s_j+
\sum_jp_j\delta v_j.
\tag{17}
$$

したがってK誤差は参照先の選択を、V誤差は取り出した内容を変える。どちらが常に重要かは決まらない。KIVI[16]のような非対称性の知見を基礎としつつ、当該モデルの$p_j$と$v_j-o$を測る必要がある。

Attention出力に基づく配分は、単独の再構成MSEを最適化するよりもモデルの機能に近い。しかし出力射影・残差・FFN以降の感度がさらに掛かるため、Attention局所の改善だけでは最終logitの改善を保証しない。

# 6. 多表現量子化と共有QJLの限界

## 6.1 consensusの利得は誤差相関で決まる

$e_r=\hat k_r-k$とすると、

$$
\sum_r\alpha_r\|e_r\|^2-
\left\|\sum_r\alpha_re_r\right\|^2
=\frac12\sum_{r,s}\alpha_r\alpha_s\|e_r-e_s\|^2\ge0.
\tag{18}
$$

合成誤差は個別誤差の加重平均以下だが、最良の単独表現以下とは限らない。同じ分散$\sigma^2$と相関$\rho$を持つ中心化スコア誤差の3等重み平均なら、分散は$\sigma^2(1+2\rho)/3$となる。Trialityから誤差独立性は導かれない。

さらに対称な量子化器では、$(-R)^\top\mathcal Q(-Rx)=R^\top\mathcal Q(Rx)$となり得る。Spinの符号を正しく保持しても、新しい量子化再構成が増えるとは限らない。厳密Triality、generic SO(8)三表現、高精度single-viewを同じ資源上限で比較することが必要である[12–18]。

## 6.2 QJLの条件付き平均・分散

固定した$q,e$が独立Gaussian行列$G$と独立であるとする。$a=q^\top e$、$Z$を式(6)のβを除いた補正項とすると、

$$
\mathbb E Z=a,\qquad
\sigma^2=\operatorname{Var}Z
=\frac{(\pi/2)\|e\|^2\|q\|^2-a^2}{m}.
\tag{19}
$$

最終raw scoreの誤差は$-(1-\beta)a+\beta(Z-a)$なので、

$$
\operatorname{MSE}
=(1-\beta)^2a^2+\beta^2\sigma^2.
\tag{20}
$$

不偏性は一般にはβ=1に限られる。βを縮めると偏りを残す代わりに分散を減らせる。$G$を使って生成・学習された$q,e$では独立性が成立しない可能性があり、固定乱数の自己回帰生成へ式(19)をそのまま適用しない[17,18,26]。

## 6.3 Fisher縮小補正

全Keyの欠損スコアを$a$、QJLノイズを$\xi$、共分散を$\Sigma$とすると、局所近似の下で

$$
\mathbb E D_{\rm KL}\simeq\frac12\left[
(1-\beta)^2a^\top H_pa+
\beta^2\operatorname{tr}(H_p\Sigma)\right].
\tag{21}
$$

分母が正なら、その最小化解は

$$
\beta_*=
\frac{a^\top H_pa}
{a^\top H_pa+\operatorname{tr}(H_p\Sigma)}.
\tag{22}
$$

これは「不偏であること」より「分布を変えないこと」を優先した縮小推定である。共有$G$を使うKey間のノイズは相関するため、$\Sigma$を独立な対角行列と無条件に置換しない。今回の数値例はoracle残差を使った合成fixtureであり、実LLMの適切なβを決めたものではない。

# 7. LLM全体への寄与を結ぶ解析

## 7.1 層ごとの誤差から最終表現へ

基準ブロックを$f_\ell$、候補ブロックを$\hat f_\ell$とし、共通入力近傍で$\|\hat f_\ell(x)-f_\ell(x)\|\le\eta_\ell$、$f_\ell$のLipschitz定数を$L_\ell$とする。両軌道を含む有界領域でこの条件が成り立てば、

$$
\|\hat h_L-h_L\|
\le \sum_{\ell=0}^{L-1}
\eta_\ell\prod_{j=\ell+1}^{L-1}L_j
+\|\hat h_0-h_0\|\prod_{j=0}^{L-1}L_j.
\tag{23}
$$

残差ブロック$f_\ell(x)=x+F_\ell(x)$では、十分な上界として$L_\ell\le1+L(F_\ell)$を使える。式(23)は逐次の三角不等式から得られる条件付き上界であり、実LLMで小さい値になるとは限らない。Attentionや正規化を含む全ブロックの感度を評価する必要がある[19,20,25]。

元モデルへの忠実度を評価するなら$\eta_\ell$にアダプターによる関数変更と量子化誤差の両方を含める。同じ学習済みアダプターのcodecのみを比較するなら、その基準関数を教師側にも置く。二つの基準を混ぜない。

## 7.2 語彙logitと下流感度

最終logitを$z=W_U N(h_L)$とし、$N$が対象領域で$L_N$-Lipschitzなら、

$$
\|\hat z-z\|\le\|W_U\|_2L_N\|\hat h_L-h_L\|.
\tag{24}
$$

RMSNormの分母下限や学習済みgainを無視して$L_N=1$とはしない。局所線形化では、Attention出力の摂動$\delta o_\ell$からlogitへのヤコビアン$J_{\ell\to z}$を使い、

$$
\delta z\simeq\sum_\ell J_{\ell\to z}\delta o_\ell,
\qquad
D_{\rm KL}^{\rm vocab}\simeq
\frac12\delta z^\top H_{p^{\rm vocab}}\delta z.
\tag{25}
$$

層ごとの重要度行列は$M_\ell=J_{\ell\to z}^\top H_{p^{\rm vocab}}J_{\ell\to z}$となる。ただし式(25)を展開すると層間交差項も現れる。誤差が独立・中心化されている等の根拠なしに交差項を捨てると、改善を過大評価し得る。

本方式のLLM全体への第一の寄与仮説は、「重要な方向・層で$\eta_\ell$または$\delta o_\ell^\top M_\ell\delta o_\ell$を下げられるか」である。全層で一様にMSEを最小化する必要はなく、同じ歪みでも後続ネットワークが増幅する方向を優先できる。

## 7.3 次トークン分布・PPL・argmax

固定文脈におけるlogit差を$\Delta z$、その最大値と最小値の差を$r$とする。$\psi(t)=\log\mathbb E_p e^{t\Delta z}$と置くと、$\psi^{\prime\prime}(t)$は指数傾斜分布下の分散で$r^2/4$以下である。式(13)は$\int_0^1(1-t)\psi^{\prime\prime}(t)\,dt$に等しいので、

$$
D_{\rm KL}(p^{\rm vocab}\|\hat p^{\rm vocab})\le r^2/8.
\tag{26}
$$

$\|\Delta z\|_\infty\le\delta$なら$r\le2\delta$で、KLは$\delta^2/2$以下となる。正解ラベルに対するNLL差の絶対値は$r$以下である。教師の最大・第2logit間marginが$r$より大きければ、argmaxは変わらない。

PPLは平均NLLの指数なので、相対PPL増加1%以内という目標は、同じ評価条件で$\Delta\overline{\mathrm{NLL}}\le\log(1.01)\simeq0.00995$ nat/tokenに対応する。ただし教師とのKLと正解ラベルに対するNLLは異なる。教師をよく模倣しても教師以上に正しくなる保証はない[11]。

## 7.4 自己回帰列全体

共通supportと有限長$T$について、基準生成分布$P$と候補$Q$はchain ruleにより

$$
D_{\rm KL}(P_{1:T}\|Q_{1:T})
=\sum_{t=1}^T\mathbb E_{x_{<t}\sim P}
D_{\rm KL}(P_t(\cdot|x_{<t})\|Q_t(\cdot|x_{<t})).
\tag{27}
$$

各項が$\epsilon_t$以下なら、全変動距離はPinsker不等式により$\sqrt{\sum_t\epsilon_t/2}$以下となる。これは長い生成で局所誤差の管理が重要な理由を与えるが、上界は長さとともに緩む。

一般のテストコーパス上のteacher-forcing KLは、式(27)の「教師が生成したprefixに関する期待値」と同一ではない。今回の合成実験はコーパスprefixで測っており、生成列全体のKL上界を実証したとはしない。greedy decoding、top-k/top-pの切断、ツール呼出し、停止条件の変更も別途評価する。

## 7.5 再帰層を変更しなくても影響は伝わる

Gated Delta型core[22]を例に、正規化された$k_t$と固定された入力・gateの下では状態誤差は

$$
E_t=\lambda_t(I-\beta_t k_tk_t^\top)E_{t-1}+\Xi_t
\tag{28}
$$

と書ける。$0\le\lambda_t\le1$、$0\le\beta_t\le2$なら係数の作用素ノルムは$\lambda_t$以下である。しかしQアダプターで上流表現が変われば、$k_t,v_t,\lambda_t,\beta_t$も変わり、追加の入力摂動項が生じる。したがって48再帰層をコード上変更しないことは、モデル全体の出力が変わらないことを意味しない。層順の再captureと最終logit評価を要求する。

## 7.6 期待される寄与と、期待してはいけないこと

期待の第一段階は、低bit環境で元モデルの参照・語彙分布を守ることである。第二段階は、独立課題による学習を通じた入力依存参照の改善である。第三段階は、低メモリ化がRAM退避や帯域律速を軽減する実装上の利得である。

これらから、新たな知識の自動獲得、ハルシネーションの消失、全言語での品質改善、推論能力の普遍的拡大は導けない。また通常のdense Attentionの系列長に関する二次演算量は残る。IO削減の効果[23]は、増加した三経路演算・変換・decodeコストと比較して初めて判定できる。

# 8. PyTorch PoCと実験方法

## 8.1 実装範囲

PoCは、正のClifford代数、同一Spin元の三表現、Q制御器、正規化Hadamard、2–8bit符号のbyte packing、GQA Attention、共有QJL、タイルonline softmaxを実装する。CUDAを指定したとき利用不能ならエラーで止まり、CPUへ無言で切り替えない。

量子化器は、group 8・対称absmax・$2^b-1$水準の**参照量子化器**である。2bitでは3水準になる。Prism PQ2_0/PTQ1_0、TheTom Turbo2–6、TurboQuantのLloyd–Max codebookをbyte互換に再実装したものではない。FP64検証ではscaleもFP64、学習実験ではscaleはFP32で保存する。低bit符号だけを数えてメモリ削減を主張しない。

タイル版はPyTorchの演算を組み合わせた参照経路であり、融合CUDA kernelではない。推論時は巨大score行列を避けるが、autogradをそのまま使う学習では中間状態が残る。FlashAttention相当のメモリ効率的backwardを実装済みとはしない。最適化されたSDPAを基準として別途microbenchmarkできる。

## 8.2 数理・数値試験

整数相当のClifford反交換、chirality、同一liftのTriality不変式、2π/4π、Q変換のノルム・det、ゼロ点gradcheck、接空間Gram、Lie代数rank、byte packing round-trip、GQA、非連続入力、全mask行、タイル・密経路の出力と勾配を検査した。

さらに、固定$q,e$に対する20,000 Gaussian sketchによるQJLの条件付き平均・分散、Fisher縮小係数の独立sketch分割、Fisher補償部分空間、有限logit差のKL上界、小さな残差ネットワークのヤコビアン伝播を調べた。これらの確率反復は独立な実LLM学習runの代用ではない。

## 8.3 小規模自己回帰Transformer

外部重み・外部データを取得せず、3層・4Qヘッド・2KVヘッド・head幅16・hidden幅64・語彙24のTransformerを学習した。RMSNorm、通常RoPE、sigmoid出力gate、FFN、残差接続、語彙headを含む。総パラメーター104,884のうち追加アダプターは2,388で、基準モデルで使用する重みは102,496である。これは再帰48層を持つBonsaiの縮小コピーではない。

課題は8記号のランダムパターンを繰り返す合成列の次記号予測で、入力長は32とした。pretraining 768、adapter training 256、selection 64、final test 128の異なるパターンを生成し、同一パターンの分割跨ぎを禁止した。各分割のhashを保存した。自然言語や一般的コード能力を評価する課題ではない。

教師はAdamW、学習率0.002、350step、batch 16で学習した。アダプターは元重みを凍結し、Adam、学習率0.008、160stepで学習した。前向き計算はpacked参照codec、量子化の逆伝播は明示的なstraight-through estimatorであり、丸めの真の微分ではない。損失は語彙KL、0.05倍の層hidden正規化MSE、0.001倍の出力投影L2を用いた。20stepごとのselection KLでcheckpointを選び、final testは学習・選択に使わない。

seedは0,1,2で、教師・データ・学習乱数が変わる。変換profileのseedは層ごとに600,601,602で固定した。したがって、全ての回転seedにわたる一般性は評価していない。generic三回転にも同じQアダプター学習予算を与えた。KV圧縮のみ、アダプターあり、非圧縮KVへのアダプター適用、高精度single-viewを分けた。

# 9. 実験結果

## 9.1 実行環境と検証結果

実測環境はLinux・Python 3.13.5・PyTorch 2.10.0+cpuである。ユーザーのWindows/Python3.12/RTX5060Ti実機とは異なる。CUDA build情報はnull、CUDA可用性はfalseであった。

| 検査 | 結果 |
|---|---:|
| 単体試験 | 25 passed、CUDA 1 skipped |
| 数値チェック | 12 / 12 |
| 限定的ソース変異 | 7 / 7検出 |
| Q変換直交性最大誤差 | $6.66\times10^{-16}$ |
| Q変換det最大誤差 | $1.78\times10^{-15}$ |
| 接空間Gram最大誤差 | $7.11\times10^{-15}$ |
| Lie代数rank | 28 |
| QJL実測分散 / 理論分散 | 1.004286 |
| QJL平均誤差 / 理論標準誤差 | $-1.1492$ |

初期RED fixture、途中で修正したFP64/FP32混在エラー、最終GREEN、故障注入ログを同梱した。7件は選択した変異に対する検出であり、網羅的なmutation scoreや製品runtimeのmutation検証ではない。

## 9.2 教師との語彙KLと次記号損失

以下は3 seedの平均であり、KLの±は**seed間標準偏差**である。PPLは平均NLLの指数、KLは教師と候補を同じテストprefixに入力して得た語彙分布の比較である。

| 方式 | 教師KL [nat/token] | NLL | PPL |
|---|---:|---:|---:|
| 教師・非圧縮KV | 0 | 0.756865 | 2.131584 |
| 単一2bit K | $0.169926\pm0.042527$ | 0.933274 | 2.542822 |
| 単一6bit K | $0.0000306\pm0.0000058$ | 0.757004 | 2.131880 |
| Triality 3×2bit K | $0.027913\pm0.009309$ | 0.786904 | 2.196586 |
| generic 3×2bit K | $0.033571\pm0.007736$ | 0.793094 | 2.210224 |
| Triality 3×2bit K＋V3bit | $0.032356\pm0.009608$ | 0.791292 | 2.206246 |
| Triality 3×2bit K＋学習Q | $0.023622\pm0.005632$ | 0.777320 | 2.175633 |
| generic 3×2bit K＋学習Q | $0.029545\pm0.005193$ | 0.784560 | 2.191443 |
| 学習Qを非圧縮KVへ適用 | $0.003564\pm0.001494$ | 0.754845 | 2.127282 |

V3bitと記した行以外のVはFP32参照である。最後の行の低いNLLは、この合成分布での観測であり、元モデル以上の一般能力を示さない。また、忠実度KLは0ではなく、関数が変化している。

TrialityのQ学習前後のKL差はseed順に0.002872、0.001511、0.008490で、平均差0.004291 nat/token、平均比で15.37%低下した。ただし、3 seedを独立単位とする探索的な対応$t$区間は$[-0.004898,0.013481]$でゼロを含む。小標本・選択的実験であり、統計的な優位性は主張しない。token数を独立標本数として信頼区間を狭めていない。

![3個の独立な教師による最終語彙KL。縦軸は対数、線は同一seedの比較を結ぶ。自然言語・GPUの結果ではない。](figures/tiny_vocab_kl.png){width=98%}

## 9.3 より重要な負の結果：高精度single-viewが優位

同じ小規模形状での**1層・1系列**当たりの実packed payloadは次のとおりである。FP32 group-scaleとVを含み、共有回転profile・scratch・allocatorは別である。

| 方式 | payload [byte] | 共有profile [byte] |
|---|---:|---:|
| 非圧縮K/V | 8,192 | 1,028（参照identity profile） |
| 単一2bit K | 4,864 | 1,028 |
| 単一6bit K | 5,376 | 1,028 |
| 三表現2bit K | 6,400 | 3,084 |
| 三表現2bit K＋V3bit | 3,200 | 3,084 |

単一6bitは三表現2bitより小さいpayloadで、教師KLも約3桁小さかった。したがって、このPoCでは三表現経路を資源効率の勝者とはできない。今回の小group・FP32 scaleはmetadata overheadが大きく、実TurboQuant/TheTom codecへの外挿もできないが、**「低bitを3本にすれば得」という前提への有効な反例**になる。

generic三回転に対するTrialityの点推定上の差も、回転seedの追加、同じ較正予算、より多くのモデルrunなしには固有利点の証拠としない。最終logitのKLが改善しても、層hidden MSEやargmax一致率が一様に改善しないrunがあり、局所指標と全体指標の区別が必要である。

## 9.4 共有QJLの縮小例

固定した9個の残差とQueryについて、5,000個のGaussian sketchから共分散を推定し、別の5,000個で評価した。Fisher式から得たβは0.513025であった。

| β | 実測Attention KL | 二次Fisher近似 |
|---|---:|---:|
| 0 | 0.003719 | 0.003821 |
| 0.513025 | 0.001839 | 0.001862 |
| 1 | 0.003667 | 0.003659 |

この条件では縮小補正が有効だった。しかしoracle残差を用いた固定fixtureであり、自己回帰LLMで同じβが適切だとは言えない。全小規模LM実験ではQJLを無効にした。QJL付き27B品質の証拠は存在しない。

# 10. 実モデルへ進むための評価設計

## 10.1 最低限の独立比較

元モデル/非圧縮KV、codecのみ、Qアダプターのみ、codec＋Qアダプターを分ける。さらにsingle-view、三表現、generic三回転、QJL β=0/較正値、実際の総容量に収まる高精度single-viewを比較する。方式選択・係数学習・最終評価を別データにする。

元モデルの参照忠実度と、独立課題の正確性・生成品質を別に測る。日本語・英語・コード・長文・ツール実行を評価し、同じNimble評価データを独立holdoutとして使わない。新Attentionの課題能力を主張するには、非圧縮KVでも同等予算のLoRA/OFT等を含む対照が必要である。

## 10.2 27Bのメモリ・学習・キャッシュ整合性

ユーザー提示の16通常Attention層、KV4、幅256なら、FP16 K/Vは64KiB/token、131,072tokenで8GiBとなる。3K＋共有Vを同じFP16で持てば16GiBで、モデル重み前にカード容量へ達する。低bit化で削減可能だが、norm/scale、再帰状態、Expert領域、scratch、driverの余裕を含めて測定する。

rank 8の小アダプターでも、32K×32K×24ヘッドのscoreをFP16で全保存すれば約48GiB/層になる。実モデル学習にはタイル化・再計算・容量上限付きcaptureが必要であり、今回のdense小規模学習コードをそのまま長文27Bへ接続しない。

Qに応じて制御量が変わるだけなら過去Kを再構築しないが、アダプターの重みを差し替えると下流K/V・再帰状態の由来が変わる。モデル・アダプター・係数・codec・数学規約・位置処理のhashを保存し、不一致cacheは再利用しない。βやαの変更も保存残差との整合性を確認する。

## 10.3 数値・実行の完成条件

FP64/FP32/FP16/BF16の変換係数監査、CPU/CUDA比較、キャンセル、OOM、cache保存復元・位置移動、未対応形式拒否を行う。監視の周期性には検出遅延があることを明記する。TheTom/Prism/Strataの型番号やHadamard順序を混同しない。

速度判定は同じ重み・入力・生成条件で行い、既存高速経路を弱めて比較しない。ユーザー仕様の生成速度5%以上、TTFT悪化5%以内、PPL増加1%以内等は**将来の採用目標**であり、今回の到達値ではない。4K/32Kの成功だけで131K障害を修正済みとしない。GPU速度、Windows復帰、無断cloud fallback禁止も製品受入試験として残る。

# 11. 限界・反証可能性・結論

本稿の理論は、数学的構成の成立から最終言語モデルの改善までを一本の保証で結ぶものではない。成立するのは、明示した条件下のノルム保存、接空間の性質、局所Fisher近似、条件付きQJL統計、誤差伝播の上界である。実モデルの入力分布、量子化の非連続性、再帰状態、長い生成、タスクの正誤に関する結論は実験が必要である。

今回のPoCは、(i) 純粋なPyTorchで構成と学習を検査できること、(ii) 小さなQアダプターが量子化後の最終logit忠実度を改善し得ること、(iii) それでも高精度single-viewが資源・精度で優位になり得ることを示した。第(ii)は3 seedの探索的観測であり、一般化や統計的優位性の確証ではない。

実用上の第一仮説は、**同じ資源制約下で、重要な方向・層の誤差を抑え、元LLMの能力を低bit・長文環境でも失いにくくできるか**である。第二仮説は、**ノルムを保つ入力依存のQuery方向変更が、量子化補償を超えて独立課題を改善するか**である。これらを分け、負の比較結果を保持することが、研究上の主張を強くする。

# 付録A. 再現手順とファイル

`triality_poc/core.py`は代数・Query・packed参照codec・共有QJL・Attention、`tiny_lm.py`は小規模モデル、`experiment.py`は学習、`validate.py`は数値検証、`benchmark.py`はCPU/CUDA演算測定を担当する。既存Strata/Prism/GGMLは変更しない。実行時にモデルやデータを外部取得せず、クラウド推論を呼ばない。

```text
python -m pytest -q
python -m triality_poc.validate --device cuda --out results/cuda_validation.json
python -m triality_poc.experiment --device cuda --out results/cuda_tiny_lm
python -m triality_poc.benchmark --device cuda --dtype float32
```

CPUでの明示実行には`--device cpu`を指定する。CUDA未検証を隠す自動CPU fallbackはない。GPU命令対応、driver、PyTorch buildはREADMEの公式導入案内で確認する。CPU参照環境は3.13.5、対象Windowsでは指定されたPython3.12の独立venvを使い、同じテストを再実行する。

同梱証跡は`evidence/final_pytest.txt`、`validation.json`、`mutations.json`、`tiny_lm/results.json`、`summary.json`である。全ファイルのSHA-256をmanifestに保存する。notebookはCPUで実行した研究用ガイドとして配布し、CUDAでの再実行は未実施と表示する。

# 付録B. 証拠レベル一覧

| レベル | 今回の状態 |
|---|---|
| 定義・恒等式・条件付き命題 | 本文に導出と条件を提示 |
| CPUの数値・勾配・単体試験 | 実行済み、ログ同梱 |
| 独立に学習した小規模モデル | 3 seed、全比較を記録 |
| CUDA実行・速度改善 | 未実施 |
| 本物のTurboQuant/Prism/TheTom codec再現 | 未実施 |
| Bonsai 27B/Strata大型Q2_0生成・PPL | 未実施 |
| Windows、サーバー復帰、長期運用 | 未実施 |
| 独立査読・優先権・一般的優位性 | 未確定 |

# 引用文献（すべてオープンアクセスで参照可能）

\begingroup
\small
\setlength{\parskip}{0.25em}

**[1]** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. 2017. *Attention Is All You Need*. arXiv:1706.03762. [OA全文・書誌](https://arxiv.org/abs/1706.03762).

**[2]** John C. Baez. 2002. *The Octonions*. arXiv:math/0105155. [OA全文・書誌](https://arxiv.org/abs/math/0105155).

**[3]** Michael Hutchinson, Charline Le Lan, Sheheryar Zaidi, Emilien Dupont, Yee Whye Teh, Hyunjik Kim. 2021. *LieTransformer: Equivariant self-attention for Lie Groups*. arXiv:2012.10885. [OA全文・書誌](https://arxiv.org/abs/2012.10885).

**[4]** Johann Brehmer, Pim de Haan, Sönke Behrends, Taco Cohen. 2023. *Geometric Algebra Transformer*. arXiv:2305.18415. [OA全文・書誌](https://arxiv.org/abs/2305.18415).

**[5]** Alex Henry, Prudhvi Raj Dachapally, Shubham Pawar, Yuxuan Chen. 2020. *Query-Key Normalization for Transformers*. arXiv:2010.04245. [OA全文・書誌](https://arxiv.org/abs/2010.04245).

**[6]** Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu. 2021. *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv:2104.09864. [OA全文・書誌](https://arxiv.org/abs/2104.09864).

**[7]** Ilya Loshchilov, Cheng-Ping Hsieh, Simeng Sun, Boris Ginsburg. 2024. *nGPT: Normalized Transformer with Representation Learning on the Hypersphere*. arXiv:2410.01131. [OA全文・書誌](https://arxiv.org/abs/2410.01131).

**[8]** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. 2021. *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685. [OA全文・書誌](https://arxiv.org/abs/2106.09685).

**[9]** Zeju Qiu, Weiyang Liu, Haiwen Feng, Yuxuan Xue, Yao Feng, Zhen Liu, Dan Zhang, Adrian Weller, Bernhard Schölkopf. 2023. *Controlling Text-to-Image Diffusion by Orthogonal Finetuning*. arXiv:2306.07280. [OA全文・書誌](https://arxiv.org/abs/2306.07280).

**[10]** Weiyang Liu, Zeju Qiu, Yao Feng, Yuliang Xiu, Yuxuan Xue, Longhui Yu, Haiwen Feng, Zhen Liu, Juyeon Heo, Songyou Peng, Yandong Wen, Michael J. Black, Adrian Weller, Bernhard Schölkopf. 2024. *Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization*. arXiv:2311.06243. [OA全文・書誌](https://arxiv.org/abs/2311.06243).

**[11]** Geoffrey Hinton, Oriol Vinyals, Jeff Dean. 2015. *Distilling the Knowledge in a Neural Network*. arXiv:1503.02531. [OA全文・書誌](https://arxiv.org/abs/1503.02531).

**[12]** Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, James Hensman. 2024. *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs*. arXiv:2404.00456. [OA全文・書誌](https://arxiv.org/abs/2404.00456).

**[13]** Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, Tijmen Blankevoort. 2025. *SpinQuant: LLM quantization with learned rotations*. arXiv:2405.16406. [OA全文・書誌](https://arxiv.org/abs/2405.16406).

**[14]** Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, Christopher De Sa. 2024. *QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks*. arXiv:2402.04396. [OA全文・書誌](https://arxiv.org/abs/2402.04396).

**[15]** Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh. 2023. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. arXiv:2210.17323. [OA全文・書誌](https://arxiv.org/abs/2210.17323).

**[16]** Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, Xia Hu. 2024. *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache*. arXiv:2402.02750. [OA全文・書誌](https://arxiv.org/abs/2402.02750).

**[17]** Amir Zandieh, Majid Daliri, Insu Han. 2024. *QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead*. arXiv:2406.03482. [OA全文・書誌](https://arxiv.org/abs/2406.03482).

**[18]** Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni. 2025. *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*. arXiv:2504.19874. [OA全文・書誌](https://arxiv.org/abs/2504.19874).

**[19]** Hyunjik Kim, George Papamakarios, Andriy Mnih. 2021. *The Lipschitz Constant of Self-Attention*. arXiv:2006.04710. [OA全文・書誌](https://arxiv.org/abs/2006.04710).

**[20]** Valérie Castin, Pierre Ablin, Gabriel Peyré. 2024. *How Smooth Is Attention?*. arXiv:2312.14820. [OA全文・書誌](https://arxiv.org/abs/2312.14820).

**[21]** James Martens. 2020. *New insights and perspectives on the natural gradient method*. arXiv:1412.1193. [OA全文・書誌](https://arxiv.org/abs/1412.1193).

**[22]** Songlin Yang, Jan Kautz, Ali Hatamizadeh. 2025. *Gated Delta Networks: Improving Mamba2 with Delta Rule*. arXiv:2412.06464. [OA全文・書誌](https://arxiv.org/abs/2412.06464).

**[23]** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré. 2022. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. arXiv:2205.14135. [OA全文・書誌](https://arxiv.org/abs/2205.14135).

**[24]** Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, Sumit Sanghai. 2023. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*. arXiv:2305.13245. [OA全文・書誌](https://arxiv.org/abs/2305.13245).

**[25]** Nikolay Yudin, Sergei Kudriashov, Alexander Gaponov, Maxim Rakhuba. 2026. *Pay Attention to Attention Distribution: A New Local Lipschitz Bound for Transformers*. arXiv:2507.07814v2. [OA全文・書誌](https://arxiv.org/abs/2507.07814v2).

**[26]** Paolo D'Alberto. 2026. *Statistical Inference and Quality Measures of KV Cache Quantisations Inspired by TurboQuant*. arXiv:2605.08114. [OA全文・書誌](https://arxiv.org/abs/2605.08114).


**ソフトウェア文書:** [PyTorch SDPA](https://docs.pytorch.org/docs/2.13/generated/torch.nn.functional.scaled_dot_product_attention.html)、[公式インストール](https://pytorch.org/get-started/locally/)、[過去バージョン](https://pytorch.org/get-started/previous-versions/)。参照日: 2026-10-23。PoCで実行した版は2.10.0+cpuであり、公開stable文書の版とは区別する。

\endgroup
