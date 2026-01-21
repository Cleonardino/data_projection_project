\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{rapport}[2026/01/21 Classe de rapport simplifiée]

\LoadClass[a4paper,12pt]{article}

% -------------------- Packages --------------------
\RequirePackage[T1]{fontenc}
\RequirePackage[french]{babel}
\RequirePackage[utf8]{inputenc}  % UTF-8
\RequirePackage{mathtools}
\RequirePackage{siunitx}
\RequirePackage{float}
\RequirePackage{graphicx}
\RequirePackage[left=2.5cm,right=2.5cm,top=1.5cm,bottom=3.5cm]{geometry}
\RequirePackage{fancyhdr}
\RequirePackage[colorlinks=true,allcolors=blue,breaklinks]{hyperref}
\RequirePackage{url}
\RequirePackage{nomencl}

% -------------------- Caption fix pour TL2025 --------------------
\RequirePackage[justification=centering]{caption}[2023/08/05]
\RequirePackage{subcaption}[2023/07/28]

% Neutralisation du message fatal dans tocbasic (TeX Live 2025)
\makeatletter
\@ifpackageloaded{tocbasic}{
  \csdef{tocbasic@caption@disable}{}
}{}
\makeatother

% -------------------- Métadonnées --------------------
\newcommand{\cours}{}
\newcommand{\sujet}{}
\newcommand{\titre}{}
\newcommand{\eleves}{}
\newcommand{\unif}{}
\newcommand{\logo}{}

\newcommand{\setcours}[1]{\def\cours{#1}}
\newcommand{\setsujet}[1]{\def\sujet{#1}}
\newcommand{\settitre}[1]{\def\titre{#1}}
\newcommand{\seteleves}[1]{\def\eleves{#1}}
\newcommand{\setunif}[1]{\def\unif{#1}}
\newcommand{\setlogo}[1]{\def\logo{#1}}

% -------------------- Mise en page --------------------
\newcommand{\fairemarges}{
  \makenomenclature
  \pagestyle{fancy}
  \fancyheadoffset{1cm}
  \setlength{\headheight}{2cm}
  \lhead{\includegraphics[width=2cm]{\logo}}
  % \chead{Rapport d'Apprentissage Automatique}
  \rhead{\nouppercase{\leftmark}}
  \cfoot{\thepage}
}

% -------------------- Page de garde --------------------
\newcommand{\fairepagedegarde}{
\begin{titlepage}
  \thispagestyle{empty} % PAS de header/footer ici
  \centering

  % Logo
  \ifx\logo\empty
  \else
    \includegraphics[width=0.6\textwidth]{\logo}
  \fi

  \vspace{2cm}

  % Université
  {\scshape\Large \unif \par}
  \vspace{1cm}

  % Cours et sujet
  {\scshape\LARGE \cours \par}
  {\scshape\LARGE \sujet \par}

  \vspace{0.5cm}

  % Titre
  \rule{\linewidth}{0.2mm}\par
  {\huge\bfseries \titre \par}
  \rule{\linewidth}{0.2mm}\par

  \vfill

  % Étudiants
  {\large \eleves \par}
  \vspace{1cm}

  % Date
  {\large 21 janvier 2026 \par}
\end{titlepage}
}

% -------------------- Table des matières --------------------
\newcommand{\tabledematieres}{
  \tableofcontents
  \newpage
}

% -------------------- Figure --------------------
\newcommand{\insererfigure}[4]{
\begin{figure}[ht]
  \centering
  \includegraphics[height=#2]{#1}
  \caption{#3}
  \label{fig:#4}
\end{figure}
}
