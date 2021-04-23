# -*- coding:utf-8 -*-
import numpy as np

from mintegpy.minterpy.utils import gamma_lp, generate_points, leja_ordered_values


# TODO lowercase
# TODO more clever way?
def checkAndCreateTreeElement(tree, ElemName):
    if ElemName not in tree:
        tree[ElemName] = TreeElement()
    return tree


class TreeElement:
    # prevent dynamic attribute assignment (-> safe memory)
    __slots__ = ['split', 'parent', 'child', 'length', 'pro_number', 'project', 'depth']

    def __init__(self, split=None, parent=None, child=None, length=2, pro_number=None, project=None, depth=0):

        # default arguments must not be mutable
        if split is None:
            split = [-1, -1]
        if parent is None:
            parent = [-1, -1]
        if child is None:
            child = [-1, -1]
        if pro_number is None:
            pro_number = []
        if project is None:
            project = []

        # TODO copy really required?
        self.split = np.array(split).copy()
        self.parent = np.array(parent).copy()
        self.child = np.array(child).copy()
        self.length = length
        self.pro_number = pro_number
        self.project = project
        self.depth = depth


class MultiIndicesTree:

    # prevent dynamic attribute assignment (-> safe memory)
    __slots__ = ['M', 'K', 'lpDegree', 'tree', 'PP', 'GP','Gamma','point_kind','optimal','optimalPP','split']

    # TODO rename argument K -> n
    def __init__(self, M=3, K=3, lpDegree=1.7,point_kind = 'leja',optimal = False,split=None):  # TODO 1.7?
        # assert M > 1, "M must be larger than 1" # FIXME

        self.M = M
        self.K = K
        self.lpDegree = lpDegree
        self.point_kind = point_kind
        self.optimal = optimal


        self.tree = None
        self.PP = None
        self.GP = None
        self.__build_Gamma()
        self.buildTree()
        self.buildPoints(split)


    def __build_Gamma(self):
        Gamma = np.zeros((self.M, 1))
        Gamma = (self.__Gamma_lp(self.M, self.K, Gamma, Gamma.copy(), self.lpDegree))
        self.Gamma = Gamma.astype(np.int)

    def __build_points(self,split):
        if split is None:
            if self.K%2:
                #print(self.K)
                self.split = np.int64(np.floor((self.K-1)/2))
            else:
                self.split = np.int64(np.floor((self.K-1)/2)+1)
        else:
            self.split = split
        #print("split __build_points",split)
        #print("split",self.split)
        points = generate_points(n = self.K,kind = self.point_kind,dim = self.M,split=self.split)
        #print("build points",points)
        #print("build points shape",points.shape)
        N = self.Gamma.shape[1]
        Points = np.zeros((self.M, self.K + 1))
        for i in range(self.M):
            Points[i,:] = (-1) ** (i + 1) * points

        PP = self.__gen_points_lp(self.M, N, Points, self.Gamma)
        self.PP = PP
        self.GP = Points

    def __build_points_optimal(self):
        points = generate_points(self.K,self.point_kind,split = self.split)[0]
        #print("points",points)
        N = self.Gamma.shape[1]
        init_points = (-1)**self.M * points
        #print("init_points",init_points)

        opt_points = np.zeros((self.M,N))
        #print("opt_points.shape",opt_points.shape)
        #print("self.Gamma.shape",self.Gamma.shape)
        for i in np.arange(N):
            #print("self.Gamma[self.M,%d]"%i,self.Gamma[self.M-1,i])
            opt_points[self.M-1,i] = init_points[self.Gamma[self.M-1,i]]

        #print("opt_points",opt_points)
        #print("self.tree[(1,1)]",self.tree[(1,1)].split)
        self.optimalPP = self.__UR_grid(opt_points,self.Gamma,self.M-1,N,1,1)
        #print("self.optimalPP ",self.optimalPP )
        #print("self.optimalPP  shape",self.optimalPP.shape)

    def __UR_grid(self,opt_points,Gamma,m,N,I,J):
        #print("N",N)
        #print("m",m)
        N0, N1 = self.tree[(I,J)].split
        #print("N0",N0)
        #print("N1", N1)
        temp_n = Gamma[m-1,N0-1] #shift N0?
        #print("temp_n",temp_n)

        points = generate_points(temp_n,self.point_kind,split = self.split)[0]
        init_points = (-1)**m * points
        for i in np.arange(N0):
            opt_points[m-1,i] = init_points[Gamma[m-1,i]]

        opt_points1 =opt_points[:,:N0]
        #print("opt_points1",opt_points1)
        opt_points2 =opt_points[:,N0:]
        #print("opt_points2",opt_points2)

        #Constant shift in opt_points2 ?!

        Gamma1 = Gamma[:,:N0]
        Gamma2 = Gamma[:,N0:N]

        I1=I+1
        I2=I+1
        J1,J2=self.tree[(I,J)].child


        if J1 > 0 and J2 > 0 and m>1:
            out = np.hstack((self.__UR_grid(opt_points1,Gamma1,m-1,N0,I1,J1),self.__UR_grid(opt_points2,Gamma2,m,N1,I2,J2)))
        elif J1 > 0 and J2 > 0 and m==1:
            out = np.hstack((opt_points1,self.__UR_grid(opt_points2,Gamma2,m,N1,I2,J2)))
        elif J1 > 0 and J2 == 0 and m>1:
            out = np.hstack((self.__UR_grid(opt_points1,Gamma1,m-1,N0,I1,J1),opt_points2))
        else:
            out=opt_points

        return out




    def buildPoints(self,split = None):
        """
        build points according to point_kind
        """
        if self.optimal:
            self.__build_points_optimal()
            self.__build_points(split = split)
        else:
            self.__build_points(split = split)

    # TODO lowercase
    def buildTree(self):
        # Produce the list of all multi-indices
        m = self.M
        n = self.K
        p = self.lpDegree
        tree = {}
        checkAndCreateTreeElement(tree, (1, 1))
        tree[(1, 1)].depth = 0

        tree = self.__gen_tree_lp(m, self.Gamma, tree, 1, 1)

        tree = self.__pro_lp(m, n, p, self.Gamma, tree, 1, 1)
        self.tree = tree



    def __gen_points_lp(self, m, N, Points, Gamma):
        PP = np.zeros((m, N))
        for i in range(N):
            for j in range(m):
                PP[j, i] = Points[j, int(Gamma[j, i])]

        return PP

    def __gen_tree_lp(self, m, Gamma, tree, I, J):
        """
        Generates tree with index properties...
        child [i,j] is J of children of [N0,N1] splitting
        parent [i,1] or [i,2] gives parent J and 1,2 specifies of [N0,N1]
        splitting
        """

        kkkk, N = Gamma.shape
        depth = tree[(1, 1)].depth
        for i in range(N):
            if abs(Gamma[m - 1, i] - Gamma[m - 1, 0]) == 1:
                N0 = i
                N1 = N - N0
                break
            N0 = N
            N1 = 0
        tree = checkAndCreateTreeElement(tree, (I, J))
        # FIXME Local variable 'N0' might be referenced before assignment
        tree[(I, J)].split = np.array([N0, N1]).copy()

        if I > depth:
            tree = checkAndCreateTreeElement(tree, (I, 1))
            tree[(I, 1)].length = 0
            tree[(1, 1)].depth = I

        k = tree[(I, 1)].length + 1

        if m > 2 and N0 > 1:
            tree = checkAndCreateTreeElement(tree, (I, 1))
            tree = checkAndCreateTreeElement(tree, (I, J))
            tree = checkAndCreateTreeElement(tree, (I + 1, k))

            tree[(I, 1)].length = k
            tree[(I, J)].child[0] = k
            tree[(I + 1, k)].parent = np.array([J, 1]).copy()

            Gamma0 = Gamma[:, 0:N0]
            tree = self.__gen_tree_lp(m - 1, Gamma0, tree, I + 1, k)
            k = k + 1
        elif m == 2 and N0 > 1:
            tree = checkAndCreateTreeElement(tree, (I, 1))
            tree = checkAndCreateTreeElement(tree, (I, J))
            tree = checkAndCreateTreeElement(tree, (I + 1, k))

            tree[(I, 1)].length = k
            tree[(I, J)].child[0] = k
            tree[(I + 1, k)].parent = np.array([J, 1]).copy()

            tree[(I + 1, k)].split = np.array([N0, 0]).copy()
            tree[(I + 1, k)].child = np.array([0, 0]).copy()
            k = k + 1
        else:
            tree[(I, J)].child[0] = 0

        if N1 > 1:
            tree = checkAndCreateTreeElement(tree, (I, 1))
            tree = checkAndCreateTreeElement(tree, (I, J))
            tree = checkAndCreateTreeElement(tree, (I + 1, k))

            tree[(I, 1)].length = k
            tree[(I, J)].child[1] = k
            tree[(I + 1, k)].parent = np.array([J, 2]).copy()

            Gamma1 = Gamma[:, N0:N]
            tree = self.__gen_tree_lp(m, Gamma1, tree, I + 1, k)
            out = tree.copy()  # FIXME return? copy?
        else:
            tree[(I, J)].child[1] = 0

        return tree

    def __Gamma_lp(self, m, n, gamma, gamma2, p):
        return gamma_lp(m, n, gamma, gamma2, p)

    def __pro_lp(self, m, n, p, Gamma, tree, I, J):
        N0 = tree[(I, J)].split[0]
        N1 = tree[(I, J)].split[1]

        Gamma0 = Gamma[:, 0:N0].copy()
        Gamma1 = Gamma[:, N0:N0 + N1].copy()
        Project = {}
        Project[(1, 1)] = [0, 0, 0]

        count = 1
        I0 = I
        J0 = J
        d = tree[(I0, J0)].split[0]
        S = np.array([d])

        if (tree[(I0, J0)].split[1] > 0):
            J0 = tree[(I0, J0)].child[1]
            I0 = I0 + 1

            for i in range(d):
                if (J0 > 0 and tree[(I, J)].split[1] > 0):
                    count = count + 1
                    S = np.insert(S, i + 1, tree[(I0, J0)].split[0])
                    # S = [S, tree[(I0,J0)].split[0]]
                else:
                    break

                J0 = tree[(I0, J0)].child[1]
                I0 = I0 + 1

        tree[(I, J)].pro_number = count - 1
        k1 = N1

        for i in range(count - 1):  # BUG? range(count-2)
            Gamma0[m - 1, :] = Gamma0[m - 1, :] + 1
            split1 = S[i]
            split2 = S[i + 1]
            if (split1 > split2):
                Pro = np.zeros((3 + split1 - split2))
                l = 0
                for j in range(split1):
                    norm = np.linalg.norm(Gamma0[:, j], p)
                    if (norm > n):
                        l += 1
                        dbg = l + 3 - 1
                        dbg2 = len(Pro)
                        Pro[l + 3 - 1] = j + 1  # BUG? Pro[.] = j+1 in case of Matlab implementation
                Pro[0] = split1 - split2
                Pro[1] = split1
                Pro[2] = split2
                Project[(1, i + 1)] = Pro

            elif (split1 == split2):
                Project[(1, i + 1)] = [0, split1, split2]

            Gamma0 = Gamma1[:, 0:split2]
            Gamma1 = Gamma1[:, split2:k1]
            k1 -= split2

        tree[(I, J)].project = Project

        if (m > 2 and count > 1):
            tree = self.__pro_lp(m - 1, n, p, Gamma[:, 0: N0], tree, I + 1, tree[(I, J)].child[0])
            out = self.__pro_lp(m, n, p, Gamma[:, N0:], tree, I + 1, tree[(I, J)].child[1])
        elif (m > 2 and count == 1):
            out = self.__pro_lp(m - 1, n, p, Gamma[:, 0:N0], tree, I + 1, tree[(I, J)].child[0])
        elif (m == 2 and (count - 1) > 1):
            out = self.__pro_lp(m, n, p, Gamma[:, N0:], tree, I + 1, tree[(I, J)].child[1])  # Replaced N0 + 1 by N0
        else:
            out = tree

        return out

    def eval_lp(self, x, C, m, n, N, gamma, Points, p, I, J):

        # FIXME Local variable 'gamma1' might be referenced before assignment
        # FIXME Local variable 'N0' might be referenced before assignment
        if m > 1 and J > 0:
            N0 = self.tree[(I, J)].split[0]
            N1 = self.tree[(I, J)].split[1]
            gamma1 = gamma
            gamma1[m - 1] = gamma1[m - 1] + 1
        elif m == 1:
            N0 = 1
            N1 = N - 1

        # TODO values for gamma1, N0, N1

        if N > 0:

            # N0, N1.. number of elements running from 0 to ... N-1
            C0 = C[0:N0]
            C1 = C[N0:N]

            if N0 > 0 and N1 > 1 and m > 2:  # N0>0 && N1>1 && m>2
                tree_child1 = self.tree[(I, J)].child[0]  # child 1
                tree_child2 = self.tree[(I, J)].child[1]  # child 2
                o1 = self.eval_lp(x, C0, m - 1, n, N0, gamma1.copy(), Points, p, I + 1, tree_child1)
                o2 = x[m - 1] - Points[m - 1, int(gamma1[m - 1]) - 1]
                o3 = self.eval_lp(x, C1, m, n - 1, N1, gamma1.copy(), Points, p, I + 1, tree_child2)
                out = o1 + o2 * o3

            elif N0 > 0 and N1 == 1 and m > 2:  # N0>0 && N1==1 && m>2
                tree_child1 = self.tree[(I, J)].child[0]  # child 1
                out = self.eval_lp(x, C0, m - 1, n, N0, gamma1.copy(), Points, p, I + 1, tree_child1) + (
                        x[m - 1] - Points[m - 1, int(gamma1[m - 1]) - 1]) * C1[0]
            elif N0 > 0 and N1 == 0 and m > 2:  # N0>0 && N1==0 && m>2
                tree_child1 = self.tree[(I, J)].child[0]  # child 1
                out = self.eval_lp(x, C0, m - 1, n, N0, gamma1.copy(), Points, p, I + 1, tree_child1)
            elif N0 > 0 and N1 > 1 and m == 2:  # N0>0 && N1>1 && m ==2
                oneD = self.eval_lp(x, C0, 1, n, N0, np.array([0]), Points, p, 0, 0)
                tree_child2 = self.tree[(I, J)].child[1]
                o2 = (x[m - 1] - Points[m - 1, int(gamma1[m - 1] - 1)])
                o3 = self.eval_lp(x, C1, m, n - 1, N1, gamma1.copy(), Points, p, I + 1, tree_child2)
                out = oneD + o2 * o3
            elif N0 > 0 and N1 == 1 and m == 2:  # N0>0 && N1==1 && m ==2
                oneD = self.eval_lp(x, C0, 1, n, N0, np.array([0]), Points, p, 0, 0)
                o2 = (x[m - 1] - Points[m - 1, int(gamma1[m - 1]) - 1])
                o3 = C1[0]
                out = oneD + o2 * o3
            elif N0 > 0 and N1 == 0 and m == 2:  # elseif N0>0 && N1==0 && m ==2
                oneD = self.eval_lp(x, C0, 1, n, N0, np.array([0]), Points, p, 0, 0)
                out = oneD
            elif m == 1 and N1 > 1:  # elseif m ==1 && N1>1
                out_1 = C0[0]
                out_2 = (x[0] - Points[0][gamma[0]])
                out_3 = self.eval_lp(x, C1, 1, n - 1, N1, np.array([gamma[0] + 1]), Points, p, 0, 0)
                out = out_1 + out_2 * out_3
            elif m == 1 and N1 == 1:  # elseif m ==1 && N1==1
                out = C0[0] + (x[0] - Points[0, gamma[0]]) * C1[0]
            elif m == 1 and N1 == 0:  # elseif m ==1 && N1==0
                out = C0[0]
            elif N0 == 0 and N1 > 0:  # elseif N0==0 && N1>0
                tree_child2 = self.tree[(I, J)].child[1]
                out = (x[m - 1] - Points[m - 1, int(gamma1[m - 1]) - 1]) * self.eval_lp(x, C1, m, n - 1, N1, gamma1,
                                                                                        Points, p, I + 1, tree_child2)
            # TODO else
        else:
            out = C[0]  # TODO default?

        return out  # FIXME Local variable 'out' might be referenced before assignment
