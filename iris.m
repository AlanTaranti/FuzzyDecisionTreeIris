%******************************************************************************
% Copyright 2016 Alan F. A. Taranti                                           *
%                                                                             *
% Licensed under the Apache License, Version 2.0 (the "License");             *
% you may not use this file except in compliance with the License.            *
% You may obtain a copy of the License at                                     *
%                                                                             *
% http://www.apache.org/licenses/LICENSE-2.0                                  *
%                                                                             *
% Unless required by applicable law or agreed to in writing, software         *
% distributed under the License is distributed on an "AS IS" BASIS,           *
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    *
% See the License for the specific language governing permissions and         *
% limitations under the License.                                              *
%                                                                             *
%******************************************************************************
%                                                                             *
%                                Iris Dataset                                 *
%                                                                             *
%                                 Abordagem:                                  *
%                      Logica Fuzzy + Arvore de Decisão                       *
%                                                                             *
%                                                                             *
%                                Alan Taranti                                 *
%                           alan.taranti@gmail.com                            *
%                       https://github.com/AlanTaranti                        *
%                                                                             *
%******************************************************************************
%                                                                             *
% Objetivo:                                                                   *
% Prototipar um classificador de três espécies de Iris:                       *
% Setosa, Versicolour e Virginica                                             *
% A partir de quatro entradas:                                                *
% Sépalo: Comprimento e Largura; Pétala: Comprimento e Largura                *
%                                                                             *
% Dataset:                                                                    *
% O conjunto de dados utilizados foi extraido do repositório de machine       *
% learning da Universidade da Califórnia Irvine                               *
% Este dataset foi criado por R.A. Fisher.                                    *
% E pode ser obtido em: https://archive.ics.uci.edu/ml/datasets/Iris          *
%                                                                             *
%******************************************************************************

% Inicialização - Limpa a tela e variáveis 
clear; clc;

% Variaveis de Controle
visualizacao = true;


% Carregar o dataset
%    O dataset usado é um pouco diferente do original,
%    todas as classes de saida, Setosa, Versicolour e Virginica
%    foram trocadas para valores numericos: 1,2 e 3 respectivamente.
fprintf('Carregando o Dataset...\n');
iris = load('iris.data');


%**************************** I - Dividir Dataset *****************************
fprintf('Dividindo o Dataset em Conjuntos de Treinamento e de Teste...\n');
n = size(iris,2)-1;              % Quantidade de caracteristicas
m = size(iris,1);                % Quantidade de elementos

random = randperm(m);            % Permuta aleatoriamente os indices do conjunto

% O dataset é dividido em dois outros conjuntos:
% Treinamento e Teste
trainRatio = 0.8;               % | O conjunto de Treinamento consistirá em 80%
                                % | do dataset original
fimTrain = floor(m*trainRatio); % | O conjunto de Teste consistirá em 20%
iniTest = fimTrain+1;           % | do dataset original

% Dividir o Dataset em conjunto de Treinamento e Teste
X = iris(random(1:fimTrain),1:n);       % Entradas do Treinamento
Y = iris(random(1:fimTrain),n+1);       % Saidas do Treinamento
Xtest = iris(random(iniTest:end),1:n);  % Entradas do Teste
Ytest = iris(random(iniTest:end),n+1);  % Saidas do Teste

m = fimTrain;                           % | Novo m: instâncias do conjunto de
                                        % | treinamento

%************** II - Gerar o Sistema de Inferencia Fuzzy Inicial **************
fprintf('Gerando o Sistema de Inferencia Fuzzy...\n');
% Inicializa o FIS (Sistema de Inferencia Fuzzy)
irisFis = newfis('irisFis');

% Nome das Entradas
columns = cell(1,n+1);
columns{1,1} = 'ComprimentoSepalo';
columns{1,2} = 'LarguraSepalo';
columns{1,3} = 'ComprimentoPetala';
columns{1,4} = 'LarguraPetala';
columns{1,5} = 'Saida';

% Gerar as entradas
for i = 1:n
    irisFis.input(i).name = columns(1,i);
end

% Gerar as funções de pertinêcia ( Membeship Functions - MF ) das entradas
%    São 3 MF para cada entrada:
%    pequeno, medio, grande
%    Todas as funções usadas são gaussianas
membFunc = cell(3,1);
membFunc{1} = 'pequeno';
membFunc{2} = 'medio';
membFunc{3} = 'grande';
for i = 1:n
    % | Define os centros das funções gaussianas
    % | utilizando os dados do conjunto de Treinamento
    minimal = min(X(:,i));                    % Menor valor - função "pequeno"
    maximum = max(X(:,i));                    % Maior valor - função "medio"
    mid = (maximum+minimal)/2;                % Valor Médio - função "grande"
    centro = [minimal,mid,maximum];           % Vetor de centros
    % Define os sigmas das funções gaussianas
    range = maximum-minimal;                  % | Intervalo da entrada do
                                              % | conjunto. Usado para definir
                                              % | o sigma
    sigma = range*0.2;                        % Sigma. Comum a todas as funções
    % Define o intervalo da entrada no FIS
    irisFis.input(i).range = [minimal maximum];
    % Define as funções de pertinêcia da entrada
    for j = 1:3
        irisFis.input(i).mf(j).name = membFunc{j};           % Nome da MF
        irisFis.input(i).mf(j).type = 'gaussmf';             % Tipo da MF
        irisFis.input(i).mf(j).params = [sigma centro(j)];   % Parametros da MF
    end

end

% Nome das Saidas
names = cell(3,1);
names{1} = 'Setosa';
names{2} = 'Versicolour';
names{3} = 'Virginica';

% Gerar as funções de pertinêcia ( Membeship Functions - MF ) das saidas
for i = 1:3
    irisFis.output(i).name = names{i};              % Nome da Saida
    irisFis.output(i).mf(1).name = names{i};        % Nome da MF
    irisFis.output(i).range = [0 1];                % Intervalo da saida
    irisFis.output(i).mf(1).type = 'trapmf';        % Tipo da MF
    irisFis.output(i).mf(1).params = [0.2 0.8 1 2]; % Parametros da MF
end


% Mostrar FIS
if visualizacao
    fprintf('Mostrando o Sistema de Inferência Fuzzy. ');
    fprintf('Aperte qualquer tecla para continuar...\n');
    pause();
    fuzzy(irisFis);
    pause();
end

%*********************** III - Gerar Árvore de Decisão ************************

fprintf('Inferindo Dataset Nítido...\n');

% Criar os rótulos pra as saidas do conjunto de Treinamento
especies = names(Y);

dm = zeros(m,n);                            % Dataset para as entrada da Árvore de Decisão
matrizInferida = cell(m,n);                 % Dataset para visualizacao textual
% Infere um atributo nítido para cada entrada
%    Essa inferência é dada através das funções de pertinêcia
%    Os atributos nítidos são definidos pela funções que retorna o
%    maior grau de pertinêcia
%    Ex: Dada uma entrada x, no qual seu grau de pertinêcia nas funções
%    "pequeno", "medio" e "grande" sejam: 0.5, 0.7 e 0.6, respectivamente
%    a entrada x é inferida como pertencente a classe "medio"
for i = 1:n
	aux = zeros(size(X,1),3);
	for j = 1:3
		aux(:,j)= gaussmf(X(:,i),irisFis.input(i).mf(j).params);
	end
	[M,I] = max(aux');
	matrizInferida(:,i) = membFunc(I');
	dm(:,i) = I';
end

% Visualização da inferência de forma textual
if visualizacao
	fprintf('Mostrando o Dataset Inferido Nítido ');
    fprintf('Aperte qualquer tecla para continuar...\n');
	pause();
	[columns ; [matrizInferida especies]] % Mostra a matriz de inferência
    fprintf('Aperte qualquer tecla para continuar...\n');
	pause();
end

% Criar a Árvore de Decisão
%    Algoritmo: CART
%    Criterio de Divisão: Gini
decisionTree = fitctree(dm,especies);


% Mostrar a Árvore de Decisão
if visualizacao
	fprintf('Mostrando o gráfico da Árvore de Decisão. ');
    fprintf('Aperte qualquer tecla para continuar...\n');
	pause();
	view(decisionTree,'mode','graph'); % Mostra a árvore de decisão (grafica)
	pause();
	fprintf('Mostrando a Árvore de Decisão em texto. ');
    fprintf('Aperte qualquer tecla para continuar...\n');
	pause();
	view(decisionTree); % Mostra a ávore de decisão (texto)
	fprintf('Aperte qualquer tecla para continuar...\n');
	pause();
end

%*********************** IV - Definir as regras dos FIS ***********************

% Antecedentes - Posição:
% n Características      FIS Entradas
% 1 ComprimentoSépalo    X 0 0 0
% 2 LarguraSépalo        0 X 0 0
% 3 ComprimentoPétala    0 0 X 0
% 4 LarguraPétala        0 0 0 X

% Antecedentes - Valores:
% n Funções        FIS Funções
% 1 pequeno        1 (é pequeno) ou -1 (não é pequeno)
% 2 medio          2 (é medio)   ou -2 (não é medio)
% 3 grande         3 (é grande)  ou -3 (não é grande)

% Consequentes - Posição:
% n Saidas         FIS Saidas
% 1 Setosa         X 0 0
% 2 Versicolour    0 X 0
% 3 Virginica      0 0 X

% Conecção - Lógica entre os antecedentes
% n Tipo
% 0 Or
% 1 And

% Regra 1
irisFis.rule(1).antecedent = [0 0 1 0];   %% se ComprimentoPétala é pequeno
irisFis.rule(1).consequent = [1 0 0];     %% então Setosa
irisFis.rule(1).connection = 1;           %% : lógica and

% Regra 2
irisFis.rule(2).antecedent = [0 0 2 -3];  %% se ComprimentoPétala é medio
                                          %% e LarguraPétala não é grande
irisFis.rule(2).consequent = [0 1 0];     %% então Versicolour
irisFis.rule(2).connection = 1;           %% : lógica and

% Regra 3
irisFis.rule(3).antecedent = [0 0 -1 3];  %% se ComprimentoPétala não é pequeno
                                          %% e LarguraPétala é grande
irisFis.rule(3).consequent = [0 0 1];     %% então Virginica
irisFis.rule(3).connection = 1;           %% : lógica and

% Regra 4
irisFis.rule(4).antecedent = [0 0 3 -3];  %% se ComprimentoPétala é grande
                                          %% e LarguraPétala não é grande
irisFis.rule(4).consequent = [0 0 1];     %% então Virginica
irisFis.rule(4).connection = 1;           %% : lógica and

% Definir o peso das Regras
for i = 1:4
	irisFis.rule(i).weight = 1;
end


%****************** V - Mostrar Sistema de Inferencia Fuzzy *******************
if visualizacao
	fprintf('Mostrando o Sistema de Inferência Fuzzy Completo. ');
    fprintf('Aperte qualquer tecla para continuar...\n');
	pause();
	fuzzy(irisFis);
	pause();
end

%***************** VI - Avaliação Sistema de Inferencia Fuzzy *****************

% Suprimir Avisos
warning ('off','all');

% Avaliar a predição no conjunto de Treinamento
% O atributo nítido é definido pelo função que gera o maior grau de pertinêcia
% na entrada
% saidaTrain = evalfis(X,irisFis);
% [M,I] = max(saidaTrain');
% saidaTrain = I';

% Avaliar a predição no conjunto de Teste
% O atributo nítido é definido pelo função que gera o maior grau de pertinêcia
% na entrada
saidaTest = evalfis(Xtest,irisFis);
[M,I] = max(saidaTest');
saidaTest = I';

% Hailitar Avisos
warning ('on','all');

%%%% Métricas de Performance
% Performance da predição no conjunto de Treinamento
fscoreT = zeros(size(names,1),1); 
prec = fscoreT;
rec = fscoreT;
acc = fscoreT;
% for i = 1:size(membFunc,1)
	% trainPredicted = saidaTrain == i;     % | Retorna todas as saidas preditas
                                          % % | iguais a i como 1
                                          
	% trainReal = Y == i;                   % | Retorna todas saidas reais
                                          % % | iguais a i como 1
	% tp = sum(trainPredicted .* trainReal);       % Verdadeiro Positivo
	% fn = sum( (trainPredicted==0) .* trainReal); % Falso Negativo
	% prec = tp/sum(trainPredicted);               % Precisão
	% rec = tp/(tp+fn);                            % Recall
	% fscoreT(i) = 2*(prec*rec)/(prec+rec);        % F-score
% end

% Performance da predição no conjunto de Teste
fscore = zeros(size(names,1),1);
for i = 1:size(names,1)
	testPredicted = saidaTest == i;       % | Retorna todas as saidas preditas
                                          % | iguais a i como 1
                                          
	testReal = Ytest == i;                % | Retorna todas saidas reais
                                          % | iguais a i como 1
	tp = sum(testPredicted .* testReal);                      % Verdadeiro Positivo
	fn = sum( (testPredicted==0) .* testReal);                % Falso Negativo
	acc(i) = sum(testPredicted == testReal)/size(testReal,1); % Acurácia
	prec(i) = tp/sum(testPredicted);                          % Precisão
	rec(i) = tp/(tp+fn);                                      % Recall
	fscore(i) = 2*(prec(i)*rec(i))/(prec(i)+rec(i));          % F-score
end

%****************************** VII - Impressão *******************************
fprintf('Resultados do Conjunto de Teste. Pressione qualquer tecla.');
fprintf('Aperte qualquer tecla para continuar...\n');
pause();
clc
for i = 1:size(names,1)
	fprintf('*** Classe: %s ***\n',names{i});
	fprintf('Acurácia: %0.2f %%\n',acc(i)*100);
	fprintf('Precisão: %0.2f %%\n',prec(i)*100);
	fprintf('Recall: %0.2f %%\n',rec(i)*100);
	fprintf('Fscore: %0.2f %%\n\n',fscore(i)*100);
end


fprintf('**********************\n');
fprintf('Exemplos de Treinamento: %d \n',fimTrain);
fprintf('Exemplos de Teste: %d \n',size(iris,1)-iniTest+1);
fprintf('Total: %d \n',size(iris,1));