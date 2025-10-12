function [Best_Fitness,Best_Pos,Convergence_curve] = OEA(N,MaxFEs,lb,ub,dim,fobj)

FEs=0;
XPosNew = zeros(N,dim);
Convergence_curve = zeros(1,MaxFEs);
lb = lb.*ones(1,dim);
ub = ub.*ones(1,dim);

XPos = lb + rand(N, dim) .* (ub - lb);

XFit = zeros(1,N);
for i = 1:N
    XFit(i) = fobj(XPos(i,:));
    FEs=FEs+1;
end

[Best_Fitness, location] = min(XFit);
Best_Pos = XPos(location,:);
Convergence_curve(1:FEs)=Best_Fitness;

while FEs < MaxFEs


    [XFit, idx] = sort(XFit, 'ascend');
    XPos = XPos(idx, :);
 
    rank_scores = ((1:N)-(N+1)/2)/(N/2);  
    k = 2 + 8*(FEs/ MaxFEs);           
    CR = 1./(1 + exp(-k*rank_scores));    
    
    F1 = 0.1 + 0.6*(1/(1+exp(-2.5*(1-2*(FEs/MaxFEs)))));
    F2 = rand;
    for i = 1:N
        jrand = floor(dim*rand + 1);
        XPosNew(i,:) = XPos(i,:);
        while true
            r1 = floor(N * (1 - sqrt(rand))) + 1;
            if r1 ~= i , break; end 
        end
        while true
            r2 = floor(N * sqrt(rand)) + 1;
            if r2 ~= i && r2 ~= r1, break; end
        end

        v1 = XPos(r2,:) - XPos(i,:);
        v2 = XPos(r1,:) - XPos(i,:);

        dot_v1_v1 = dot(v1, v1);
        
        if abs(dot_v1_v1) < 1e-20
            v2_orth = v2; 
        else
            dot_v2_v1 = dot(v2, v1);
            projection = (dot_v2_v1 / dot_v1_v1) * v1;
            v2_orth = v2 - projection;
        end

        update_vector = (F1 * v1) + (F2 * v2_orth);

        for j = 1:dim
            if rand < CR(i) || j == jrand
                XPosNew(i,j) = Best_Pos(j) + update_vector(j);
            end
        end

        flagub = XPosNew(i,:) > ub;
        flaglb = XPosNew(i,:) < lb;
        XPosNew(i, flagub) = (XPos(i, flagub) + ub(flagub)) / 2;
        XPosNew(i, flaglb) = (XPos(i, flaglb) + lb(flaglb)) / 2;
        
        if FEs >= MaxFEs, break; end
        new_XFit = fobj(XPosNew(i,:));
        FEs=FEs+1;

        if new_XFit <= XFit(i)
            XPos(i,:) = XPosNew(i,:);
            XFit(i) = new_XFit;
            if XFit(i) < Best_Fitness
                Best_Fitness = new_XFit;
                Best_Pos = XPos(i,:);
            end
        end
        Convergence_curve(FEs)=Best_Fitness;
    end
    if FEs >= MaxFEs, break; end
end
Convergence_curve=Convergence_curve(1:MaxFEs);
end