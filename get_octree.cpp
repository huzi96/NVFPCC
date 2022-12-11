#include <iostream>
#include <map>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <fstream>
#include <queue>
#include <cmath>
typedef unsigned int uint32;
typedef unsigned long uint64;
using namespace std;

template<class T>
class Coordinate
{
public:
    T vec[3];
    Coordinate(){}

    Coordinate(T x, T y, T z) 
    {vec[0] = x; vec[1] = y; vec[2] = z; }
    
    Coordinate(T x[])
    {vec[0] = x[0]; vec[1] = x[1]; vec[2] = x[2]; }

    T & operator [](int idx)
    {
        if ( idx > 2 || idx < 0) {
            fprintf(stderr, "Invalid index!\n"); exit(-1);
        }    
        return vec[idx];
    }

    const T & operator [](int idx) const
    {
        if ( idx > 2 || idx < 0) {
            fprintf(stderr, "Invalid index!\n"); exit(-1);
        }    
        return vec[idx];
    }

    Coordinate operator + (const Coordinate<T> &y)
    {
        Coordinate<T> ans(*this);
        for (int i = 0; i < 3; i++) {
            ans[i] += y[i];
        }
        return ans;
    }

    Coordinate operator - (const Coordinate<T> &y)
    {
        Coordinate<T> ans(*this);
        for (int i = 0; i < 3; i++) {
            ans[i] -= y[i];
        }
        return ans;
    }

    Coordinate operator * (T scalar)
    {
        Coordinate<T> ans(*this);
        for (int i = 0; i < 3; i++) {
            ans[i] *= scalar;
        }
        return ans;
    }

    bool operator == (const Coordinate<T> &y)
    {
        bool flag = true;
        for (int i = 0; i < 3; i++) {
            if (y[i] != vec[i]) flag = false;
        }
        return flag;
    }

    template<class T2>
    Coordinate(const Coordinate<T2> ref)
    { vec[0] = (T2)ref[0]; vec[1] = (T2)ref[1]; vec[2] = (T2)ref[2]; }

    double modulus() {
        return sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
    }
};

template<class T1, class T2>
double dot(Coordinate<T1> x, Coordinate<T2> y)
{
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}


template<class T>
ostream & operator << (ostream & out, const Coordinate<T> &x)
{
        // out << "(" << x[0] << "," << x[1] << "," << x[2] << ")";
        out << x[0] << "," << x[1] << "," << x[2];

        return out;
}

class CoordinateList
{
public:
    Coordinate<uint64>* pts;
    int npts;
    int max_npts;
    CoordinateList(int _npts, const Coordinate<uint64> *_pts);
    CoordinateList(int _max_npts);
    CoordinateList(const CoordinateList& x);
    ~CoordinateList();
    const Coordinate<uint64>& operator [] (int idx) const;
    Coordinate<uint64>& operator [] (int idx);
    const CoordinateList& append(Coordinate<uint64> x)
    {
        if (npts + 1 > max_npts) {
            fprintf(stderr, "Exceed point list limit\n");
            exit(-1);
        }
        pts[npts] = x;
        npts++;
        return *this;
    }
};

class LabelCoordinateList:public CoordinateList
{   
public:
    int *label;
    LabelCoordinateList(int _npts, const Coordinate<uint64> *_pts):CoordinateList(_npts, _pts)
    {
        label = new int[max_npts];
    }
    LabelCoordinateList(int _max_npts):CoordinateList(_max_npts)
    {
        label = new int[max_npts];
    }
    LabelCoordinateList(const CoordinateList& x):CoordinateList(x)
    {
        label = new int[max_npts];
    }
    ~LabelCoordinateList()
    {
        delete []label;
    }
    int& operator () (int idx)
    {
        if (idx >= npts)
        {
            fprintf(stderr, "Invalid label visit\n");
            exit(-1);   
        }
        return label[idx];
    }
};

class VoxelFP
{
public:
    uint64 digest;
    VoxelFP(Coordinate<uint64> p, int level=0)
    {
        if(p[0] >= (1<<18) || p[1] >= (1<<18) || p[2] >= (1<<18) || level >= 18) {
            fprintf(stderr, "Unable to digest\n");
            cerr << p << " " << level<<endl;
            exit(-1);
        }
        digest = p[0];
        digest += (p[1] << 18);
        digest += (p[2] << 36);
        digest += (((uint64)level) << 54);
    }
    bool operator < (const VoxelFP &x) const
    {
        return digest < x.digest;
    }
};

class CornerFP:public VoxelFP
{
public:
    CornerFP(Coordinate<uint64> p, int level):VoxelFP(p, level){}
    bool operator < (const CornerFP &x) const
    {
        return digest < x.digest;
    }
};

class Voxel;

class Corner
{
public:
    Voxel* neighbors[8];
    Coordinate<uint64> coord;
    Coordinate<double> normal;
    Corner(Coordinate<uint64> _coord):coord(_coord),normal(0,0,0)
    {
        for (int i = 0; i < 8; i++) {
            neighbors[i] = NULL;
        }
    }
    Corner():normal(0,0,0)
    {
        for (int i = 0; i < 8; i++) {
            neighbors[i] = NULL;
        }
        coord = Coordinate<uint64>(0, 0, 0);
    }
};

class Voxel
{
public:
    Corner corners[8];
    Voxel* neighbors[6];
    CoordinateList points;
    Coordinate<uint64> origin;
    uint64 scale;
    Voxel(Coordinate<uint64> _origin, const int npts, const CoordinateList& pts, uint64 _scale);
};

class Tree;

class CallbackFunc
{
public:
    virtual int operator()(Tree* tree) = 0;
};


class Tree
{
public:
    Voxel* current_voxel;
    Tree* children[8];
    uint32 level;
    Tree(uint32 _level=0);
    Tree* build_tree(Coordinate<uint64> origin, int init_scale, int npts, const CoordinateList &pts);
    Tree* build_neighborhood();
    void traverse(CallbackFunc *func);
    void update_neighborhood();
};


map<VoxelFP, Voxel *> global_voxel_LUT;
map<CornerFP, Corner> global_corner_LUT;


class DisplayCallback:public CallbackFunc
{
public:
    virtual int operator()(Tree* tree)
    {
        if(tree->current_voxel->scale == 1) {
            cout << tree->current_voxel->origin << endl;
            for (int i = 0; i < 8; i++) {
                cout << "\tCorner: " << tree->current_voxel->corners[i].coord <<endl; 
                // if (tree->current_voxel->neighbors[i] != NULL) {
                //     cout << "\tNeighbor: " << tree->current_voxel->neighbors[i]->origin << endl;
                // }
            }
        }
        return 0;
    }
};

class UpdateCornerCallback:public CallbackFunc
{
public: 
    virtual int operator()(Tree* tree)
    {
        Voxel* vox_ptr = tree->current_voxel;
        for (int i = 0; i < 8; i++) {
            Coordinate<uint64> coord = vox_ptr->corners[i].coord;
            int level = tree->level;
            CornerFP fp(coord, level);
            if (global_corner_LUT.find(fp) == global_corner_LUT.end()) {
                fprintf(stderr, "Corner not found in the global LUT during update.\n");
                exit(-1);
            }
            vox_ptr->corners[i] = global_corner_LUT[fp];
        }
        return 0;
    }
};

DisplayCallback display_call_back;

Coordinate<uint64> raw_pts[10000000];

Voxel::Voxel(Coordinate<uint64> _origin, const int npts, const CoordinateList& pts, uint64 _scale):origin(_origin), scale(_scale), points(pts)
{}

CoordinateList::CoordinateList(int _npts, const Coordinate<uint64> *_pts)
{
    max_npts = _npts;
    pts = new Coordinate<uint64>[max_npts];
    npts = _npts;
    for (int i = 0; i < npts; i++) {
        pts[i] = _pts[i];
    }
}

CoordinateList::CoordinateList(int _max_npts)
{
    max_npts = _max_npts;
    npts = 0;
    pts = new Coordinate<uint64>[max_npts];
}

CoordinateList::CoordinateList(const CoordinateList& x)
{
    max_npts = x.max_npts;
    pts = new Coordinate<uint64>[max_npts];
    npts = x.npts;
    for (int i = 0; i < npts; i++) {
        pts[i] = x.pts[i];
    }   
}

CoordinateList::~CoordinateList()
{
    delete[] pts;
}

const Coordinate<uint64>& CoordinateList::operator [] (int idx) const 
{ return pts[idx]; }

Coordinate<uint64>& CoordinateList::operator [] (int idx) 
{ return pts[idx]; }

void Tree::traverse(CallbackFunc *func)
{
    int ret = (*func)(this);
    if (ret != 0) return; 
    for (int i = 0; i < 8; i++) {
        if (children[i] != NULL) {
            children[i] -> traverse(func);
        }
    }
}

Tree::Tree(uint32 _level):level(_level)
{
    current_voxel = NULL;
    for (int i = 0; i < 8; i++) {
        children[i] = NULL;
    }
}

Tree* Tree::build_tree(Coordinate<uint64> origin, int init_scale, int npts, const CoordinateList &pts)
{
    if (npts == 0) {
        return NULL;
    }
    // Build the first voxel
    current_voxel = new Voxel(origin, npts, pts, init_scale);
    VoxelFP fp(origin, level);
    global_voxel_LUT[fp] = current_voxel;
    // Quadrant partitioning
    if (init_scale == 1) {
        return this;
    }
    int scale = init_scale / 2;
    int steps[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}
    };

    CoordinateList children_pts[8]={npts, npts, npts, npts, npts, npts, npts, npts};

    for (int i = 0; i < npts; i++) {
        char flag = 0;
        const Coordinate<uint64> &p = pts[i];
        if (p[0] >= origin[0] + scale) {
            flag += 1;
        }
        if (p[1] >= origin[1] + scale) {
            flag += 2;
        }
        if (p[2] >= origin[2] + scale) {
            flag += 4;
        }
        children_pts[flag].append(p);
    }
#ifdef DEBUG
    cout << endl << "Origin: " << origin << " scale: " << scale << endl;
    for (int i = 0; i < 8; i++)
    {
        cout << "Children " << i << " has " << children_pts[i].npts << " points" <<endl;
        for (int j = 0; j < min(10, children_pts[i].npts); j++) {
            cout << "\t" << children_pts[i][j] << endl;
        }
        
    }
#endif
    for (int i = 0; i < 8; i++) {
        if (children_pts[i].npts == 0) {
            // no point in this quadrant
            children[i] = NULL;
            continue;
        }
        Coordinate<uint64> new_origin(origin[0]+steps[i][0]*scale, origin[1]+steps[i][1]*scale, origin[2]+steps[i][2]*scale);
        children[i] = new Tree(level+1);
        children[i]->build_tree(new_origin, scale, children_pts[i].npts, children_pts[i]);
    }
    return this;
}

Tree* Tree::build_neighborhood()
{
    int steps[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}
    }; 
    int scale = current_voxel->scale;
    Coordinate<uint64> origin = current_voxel->origin;
    //register and update all corners
    for (int i = 0; i < 8; i++) {
        Coordinate<uint64> corner_pt(origin[0]+steps[i][0]*scale, origin[1]+steps[i][1]*scale, origin[2]+steps[i][2]*scale);
        CornerFP fp(corner_pt, level);
        Corner corner(corner_pt);
        if (global_corner_LUT.find(fp) != global_corner_LUT.end()) {
            corner = global_corner_LUT[fp];
        }
        int idx = 7 - i;
        if(corner.neighbors[idx] != NULL) {
            fprintf(stderr, "Error: the voxel neighbor of this corner has been occupied.\n");
            exit(-1); // a corner has multi-scale voxel neighbor
        }
        corner.neighbors[idx] = current_voxel;
        current_voxel->corners[i] = corner;
        global_corner_LUT[fp] = corner;
    }

    //register neighbor voxels
    int nb_steps[6][3] = {
        {-1, 0, 0}, {0, -1, 0}, {0, 0, -1},
        {1, 0, 0}, {0, 1, 0}, {0, 0, 1}
    };
    for (int i = 0; i < 6; i++) {
        int flag = 0;
        for (int j = 0; j < 3; j++) {
            if ((long)origin[j]+nb_steps[i][j]*scale < 0) flag = 1;
        }
        if (flag == 1) {
            current_voxel->neighbors[i] = NULL;
            continue;
        }
        Coordinate<uint64> nb_origin(origin[0]+nb_steps[i][0]*scale, origin[1]+nb_steps[i][1]*scale, origin[2]+nb_steps[i][2]*scale);
        VoxelFP fp(nb_origin, level);
        if (global_voxel_LUT.find(fp) == global_voxel_LUT.end()) {
            current_voxel->neighbors[i] = NULL;
        }
        else {
            current_voxel->neighbors[i] = global_voxel_LUT[fp];
        }
    }
    for (int i = 0; i < 8; i++) {
        if (children[i] != NULL) {
            children[i] -> build_neighborhood();
        }
    }
    
    return this;
}

void Tree::update_neighborhood()
{
    UpdateCornerCallback update_corner_callback;
    CallbackFunc *func = (CallbackFunc*) &update_corner_callback;
    traverse(func);
}

Coordinate<double> normal_presets[8] = {
    Coordinate<double>( 1, 1, 1),
    Coordinate<double>(-1, 1, 1),
    Coordinate<double>( 1,-1, 1),
    Coordinate<double>(-1,-1, 1),
    Coordinate<double>( 1, 1,-1),
    Coordinate<double>(-1, 1,-1),
    Coordinate<double>( 1,-1,-1),
    Coordinate<double>(-1,-1,-1),
};

void calculate_normal(Tree* root)
{
    queue<Tree *> q;
    q.push(root);
    while(!q.empty()) {
        Tree* crt = q.front();
        q.pop();

#ifdef DEBUG 
        CornerFP probe(Coordinate<uint64>(0,512,0), 1);
        if (global_corner_LUT.find(probe) != global_corner_LUT.end()) {
            cout << global_corner_LUT[probe].normal << endl;
            for (int k = 0; k < 8; k++) {
                if(global_corner_LUT[probe].neighbors[k] != NULL) cout << "1 ";
                else cout << "0 ";
            }
            cout << endl;
        }
        else {
            cout << "Not found" << endl;
        }
#endif
        // Handle all corners related to this voxel
        for (int i = 0; i < 8; i++) {
            Corner c = crt->current_voxel->corners[i];
            Coordinate<double> n(0,0,0);

#ifdef DEBUG2
            cout << "Processing " << c.coord << " at level " << crt->level << " with voxel size " << crt->current_voxel->scale << " and npts " << crt->current_voxel->points.npts << endl;
            for (int j = 0; j < 8; j++) {
                if (c.neighbors[j] != NULL) {
                    n  = n + normal_presets[j];

                    cout << "1 ";

                }
                else cout << "0 ";
            }
            cout << "Normal: " << n << endl;
#else
            for (int j = 0; j < 8; j++) {
                if (c.neighbors[j] != NULL) {
                    n  = n + normal_presets[j];
                }
            }
#endif
            c.normal = n;


            CornerFP fp(c.coord, crt->level);
            if (global_corner_LUT.find(fp) == global_corner_LUT.end()) {
                fprintf(stderr, "The current corner is not in the global LUT.\n");
                exit(-1);
            }
            Corner c_retr = global_corner_LUT[fp];
            if (!(c_retr.normal == Coordinate<double>(0,0,0)) && !(c_retr.normal == c.normal)) {
                fprintf(stderr, "The current corner's normal does not match the one in the global LUT.\n");
                cout << "Coord: " << c_retr.coord << " " << crt->level << " N: " << c_retr.normal << endl;
                for (int k = 0; k < 8; k++) {
                    if(c_retr.neighbors[k] != NULL) cout << "1 ";
                    else cout << "0 ";
                }
                cout << endl;
                
                cout << "Coord: " << c.coord << " " << crt->level << " N: " << c.normal << endl;
                for (int k = 0; k < 8; k++) {
                    if(c.neighbors[k] != NULL) cout << "1 ";
                    else cout << "0 ";
                }
                cout << endl;
                exit(-1);
            }
            crt->current_voxel->corners[i].normal = n;
            global_corner_LUT[fp].normal = n;
        }
        // Handle its children
        for (int i = 0; i < 8; i++) {
            if (crt->children[i] != NULL) {
                q.push(crt->children[i]);
            }
        }
    }
}


void get_binary_representation(Tree* root, int up_to_level, ostream& out)
{
    queue<Tree *> q;
    q.push(root);
    while(!q.empty()) {
        Tree* crt = q.front();
        q.pop();
        // Output children information       
        // And handle its children
        for (int i = 0; i < 8; i++) {
            if (crt->children[i] != NULL) {
                out << "1";
                if (crt->children[i]->level <= up_to_level) {
                    q.push(crt->children[i]);
                }
            }
            else {
                out << "0";
            }
        }
    }
}

CoordinateList origin_at_level_x(1000000);
class CollectLevelXCallback:public CallbackFunc
{
public:
    int x;
    CollectLevelXCallback(int _x):x(_x){}
    virtual int operator()(Tree* tree)
    {
        if (tree->level == x) {
            origin_at_level_x.append(tree->current_voxel->origin);
            return 1;
        }
        return 0;
    }
};

pair<bool, Coordinate<double> > queryNormalAtLevelX(Coordinate<double> query, int level)
{
    int q_step = 1 << (10 - level);
    Coordinate<uint64> quantized_query(
        (uint64)(query[0]/q_step)*q_step,
        (uint64)(query[1]/q_step)*q_step,
        (uint64)(query[2]/q_step)*q_step);
    VoxelFP fp(quantized_query, level);
    if (global_voxel_LUT.find(fp) == global_voxel_LUT.end()) {
        return make_pair(false, Coordinate<double>(0,0,0));
    }
    double lambda_x = 1 - ((double)query[0] - quantized_query[0]) / q_step;
    double lambda_y = 1 - ((double)query[1] - quantized_query[1]) / q_step;
    double lambda_z = 1 - ((double)query[2] - quantized_query[2]) / q_step;

    Voxel* voxel = global_voxel_LUT[fp];

    Coordinate<double> x_mids[4];
    x_mids[0] = voxel->corners[0].normal * lambda_x + voxel->corners[1].normal * (1-lambda_x);
    x_mids[1] = voxel->corners[2].normal * lambda_x + voxel->corners[3].normal * (1-lambda_x);
    x_mids[2] = voxel->corners[4].normal * lambda_x + voxel->corners[5].normal * (1-lambda_x);
    x_mids[3] = voxel->corners[6].normal * lambda_x + voxel->corners[7].normal * (1-lambda_x);

    Coordinate<double> y_mids[2];
    y_mids[0] = x_mids[0] * lambda_y + x_mids[1] * (1-lambda_y);
    y_mids[1] = x_mids[2] * lambda_y + x_mids[3] * (1-lambda_y);

    Coordinate<double> z_mid;
    z_mid = y_mids[0] * lambda_z + y_mids[1] * (1-lambda_z);
    return make_pair(true, z_mid);
}

Coordinate<double> queryNormal(Coordinate<double> query)
{
    for (int l = 10; l >= 0 ; l--) {
        pair<bool, Coordinate<double> > res = queryNormalAtLevelX(query, l);
        if (res.first == false) {
            continue;
        }
        Coordinate<double> z_mid = res.second;
#ifdef DEBUG2
        cout << "Voxel matched at level " << l << endl;
        cout << "Lambdas: " << lambda_x << " " << lambda_y << " " << lambda_z << endl;
        for (int i = 0; i < 8; i++) {
            cout << voxel->corners[i].coord << "   " << voxel->corners[i].normal << endl;
        }
        
#endif
        return z_mid;
    }
    fprintf(stderr, "Unable to proceed with the query.\n");
    cerr << query << endl;
    exit(-1);
}

inline bool is_solid_l10_voxel(Coordinate<uint64> p)
{
    VoxelFP fp(p, 10);
    if(global_voxel_LUT.find(fp) == global_voxel_LUT.end()) {
        return false;
    }
    return true;
}

double get_dist(Coordinate<uint64> query)
{
    Coordinate<double> O(query[0]+0.5, query[1]+0.5, query[2]+0.5);
    Coordinate<double> p = O;
    Coordinate<double> n0(0,0,0);
    int M = 1024;
    bool flag = false;
    double d = 0;
    while (M-->0) {
        // Check whether p in a solid voxel
        Coordinate<uint64> discrete_p(p);
        if (is_solid_l10_voxel(discrete_p)) {
            flag = true;
            if (M == 1024) {
                d = 0;
                break;
            }
            Coordinate<double> center_p(discrete_p[0]+0.5, discrete_p[1]+0.5, discrete_p[2]+0.5);
            Coordinate<double> n(0,0,0);
            for (int l = 10; l >= 5; l--) {
                pair<bool, Coordinate<double> > res = queryNormalAtLevelX(center_p, l);
                if (res.first == false) {
                    continue;
                }
                n = res.second;
                if (n[0]==0 && n[1]==0 && n[2]==0) {
                    continue;
                }
                break;
            }
            
            if (n[0]==0 && n[1]==0 && n[2]==0) {
                fprintf(stderr, "Error. Zero normal vector.\n");
                exit(-1);
            }
            Coordinate<double> op = O - p;
            d = abs(dot(op, n)) / abs(n.modulus());
            break;
        }
        else {
            n0 = queryNormal(p);
            n0 = n0 * (-1/n0.modulus());
            p = p + n0 * 0.25;
        }
    }
    if (flag == false) {
        cerr << "Error. Ray casting fail to reach a solid voxel." << endl;
        cerr << "Origin: " << O << endl;
        cerr << "n0: " << n0 << endl;
        cerr << "Final p: " << p << endl;
        exit(-1);
    }
    return d;
}

Voxel* globalLevel5VoxelList[1000];
int ptrGlobalLevel5VoxelList = 0;
class CollectLevel5VoxCallback:public CallbackFunc
{
public:
    virtual int operator()(Tree* tree)
    {
        if (tree->level == 5) {
            globalLevel5VoxelList[ptrGlobalLevel5VoxelList++] = tree->current_voxel;
            return 1;
        }
        return 0;
    }
};

/*
Usage example: 
    ./get_octree input_ascii.ply origins.txt subtree.txt
*/
int main(int argc, char *argv[])
{
    /* Read the point cloud */
    fstream f;
    char buffer[128];
    f.open(argv[1], ios::in);
    char buffer_seg[128];
    int npts = 0;
    double qscale = 1;
    while (true) {
        f.getline(buffer, 127);
        memcpy(buffer_seg, buffer, 14);
        buffer_seg[14] = 0;
        if (strcmp(buffer_seg, "element vertex") == 0) {
            npts = atoi(buffer+15);
        }
        buffer[10] = 0;
        if (strcmp(buffer, "end_header") == 0) {
            break;
        }
    }

    for (int i = 0; i < npts; i++) {
        f.getline(buffer, 127);
        int x, y, z, r, g, b;
        sscanf(buffer, "%d%d%d%d%d%d", &x, &y, &z, &r, &g, &b);
        raw_pts[i] = Coordinate<uint64>(x, y, z);
    }
    CoordinateList plist(npts, raw_pts);
    
    /* Build the Octree and calculate distance field */
    Tree tree;
    tree.build_tree(Coordinate<uint64>(0, 0, 0), 1024, npts, plist);
    tree.build_neighborhood();
    tree.update_neighborhood();

    /* Collect all level-5 cube origins */
    CollectLevelXCallback collect_level_x(5);
    tree.traverse(&collect_level_x);
    fstream out_f1;
    out_f1.open(argv[2], ios::out);
    for (int i = 0; i < origin_at_level_x.npts; i++) {
        out_f1 << origin_at_level_x[i] << endl;
    }
    out_f1.close();

    /* Collect shallow subtree binary representation */
    fstream out_f2;
    out_f2.open(argv[3], ios::out);
    get_binary_representation(&tree, 5, out_f2);
}
