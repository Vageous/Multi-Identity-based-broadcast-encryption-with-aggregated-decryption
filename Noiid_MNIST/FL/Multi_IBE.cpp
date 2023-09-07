#include<iostream>
#include<vector>
#include<string>
#include<NTL/ZZ.h>
using namespace std;
using namespace NTL;
#define security_parameter 64;

/*
Problem:
	the random number r in PrivateEnc is always 0;
	change the code that the random number r to be another number.
*/



class IBE {
public:
	ZZ g, p, q, n, N, g1, g2, g3, order;
	vector<ZZ>mpk1, msk, msk1, hash1, hash2, R;
	void setup(int k, int num);
	void hash(vector<string>ID);
	void L(ZZ& x);
	void ran_select(ZZ& r);
	void select(ZZ x1, ZZ x2, ZZ& s1, ZZ& s2);
	vector<ZZ> extract(vector<ZZ> msk);
	vector<ZZ> usk2(vector<ZZ> pk, vector<ZZ> sk);
	vector<vector<ZZ>> PrivateEnc(vector<ZZ> m, ZZ usk2);
	void aggregate(vector<vector<ZZ>>& C,vector<vector<ZZ>> c);
	vector<ZZ> decrypt(vector<vector<ZZ>>C, ZZ usk1,int num);
};

//Hash function: H1:{0,1}-> Z_N
// input: s (string)
// output: hash (ZZ)

void IBE::hash(vector<string> ID) {
	for (size_t i = 0; i < ID.size(); i++) {
		ZZ s1, t1, s2, t2, temp1, temp2, mod;
		mod = q * n;
		s1 = RandomBnd((ZZ)1000);
		t1 = RandomBnd(ZZ(1000));
		s2 = RandomBnd((ZZ)1000);
		t2 = RandomBnd(ZZ(1000));
		select(msk[2], msk[3], s1, s2);
		temp1 = AddMod(MulMod(s1, p, N), MulMod(t1, mod, N), N);
		temp2 = AddMod(MulMod(s2, p, N), MulMod(t2, mod, N), N);
		hash1.push_back(temp1);
		hash2.push_back(temp2);
	}
}

void IBE::L(ZZ& x) {
	x = (x - 1) / n;
}

void IBE::ran_select(ZZ& r) {
	ZZ temp1, temp2, temp3;
	temp1 = PowerMod(g1, r, N);
	temp2 = PowerMod(g2, r, N);
	temp3 = PowerMod(g3, r, N);
	while (GCD(temp1, N) != 1 || GCD(temp2, N) != 1 || GCD(temp3, N) != 1) {
		r = RandomBnd((ZZ)1000);
		temp1 = PowerMod(g1, r, N);
		temp2 = PowerMod(g2, r, N);
		temp3 = PowerMod(g3, r, N);
	}
}



vector<ZZ> IBE::usk2(vector<ZZ>pk, vector<ZZ>sk) {
	vector<ZZ>usk3;
	int len = pk.size();
	ZZ temp1 = pk[1], temp2;
	for (int i = 2; i < len; i++) {
		temp1 = MulMod(temp1, pk[i], N);
	}
	usk3.push_back(PowerMod(temp1, -sk[0], N));
	for (int k = 1; k < len - 1; k++) {
		temp1 = pk[0];
		temp2 = pk[k + 1];
		for (int i = 1; i < k; i++) {
			temp1 = MulMod(temp1, pk[i], N);
		}
		for (int j = k + 2; j < len; j++) {
			temp2 = MulMod(temp2, pk[j], N);
		}
		usk3.push_back(PowerMod(MulMod(temp1, InvMod(temp2, N), N), sk[k], N));
	}
	temp1 = pk[0];
	for (int i = 1; i < len - 1; i++) {
		temp1 = MulMod(temp1, pk[i], N);
	}
	usk3.push_back(PowerMod(temp1, sk[len - 1], N));
	return usk3;
}

void IBE::select(ZZ x1, ZZ x2, ZZ& s1, ZZ& s2) {
	ZZ temp, mod;
	mod = n * q;
	temp = AddMod(MulMod(x1, s1, mod), MulMod(x2, s2, mod), mod);
	while (GCD(temp, mod) != 1) {
		s1 = RandomBnd((ZZ)1000);
		s2 = RandomBnd((ZZ)1000);
		temp = AddMod(MulMod(x1, s1, mod), MulMod(x2, s2, mod), mod);
	}
}

/// <summary>
/// generate the public parameters and the mster key (mpk,msk)
/// input: k (int)
/// output:
/// mpk={N, g, P_pub, H_1, H_2, h_i} (ZZ)
/// msk={x_1,x_2, alpha_i} (ZZ)
/// </summary>
void IBE::setup(int k, int num) {
	RandomPrime(p, k/2, 10);
	RandomPrime(q, k/2, 10);
	ZZ item1, item2;
	item1 = 2 * p + 1;
	item2 = 2 * q + 1;
	while (MillerWitness(item1, (ZZ)100) != 0 || MillerWitness(item2, (ZZ)100) != 0) {
		RandomPrime(p, k / 2, 10);
		RandomPrime(q, k / 2, 10);
		item1 = 2 * p + 1;
		item2 = 2 * q + 1;
	}
	msk.push_back(p);
	msk.push_back(q);
	n = (2 * p + 1) * (2 * q + 1);
	N = n * n;
	order = n * p * q;
	int j = 0;
	ZZ x, temp, temp1;
	temp1 = 2;
	temp = SqrMod(temp1, N);
	//cout << GCD(temp, N) << endl;
	while (GCD(temp, N) != 1) {
		++temp1;
		temp = SqrMod(temp1, N);
	}
	g = SqrMod(temp1, N);
	//cout << g << endl;
	//cout << PowerMod(g, order, N) << endl;
	while (j < num) {
		x = RandomBnd(1000);
		if (GCD(PowerMod(g, x, N), N) == 1) {
			msk1.push_back(x);
			mpk1.push_back(PowerMod(g, x, N));
			j++;
		}
	}

	for (int i = 0; i < 2; i++) {
		msk.push_back(RandomBnd(n - 1));
	}
	g1 = PowerMod(g, msk[2] * p, N);
	g2 = PowerMod(g, msk[3] * p, N);
	g3 = PowerMod(g, p * p, N);
}


// Extract: take the user's identity ID as input, and output the user's private key using the master secret key.
//input: ID_i, msk
//output: sk_i

vector<ZZ> IBE::extract(vector<ZZ> msk) {
	vector<ZZ> usk;
	ZZ h1, h2,s1,s2,t1,t2,temp;
	ZZ mod = n * msk[1];
	ZZ inv_p = InvMod(msk[0], mod);
	for (int i = 0; i < hash1.size(); i++) {
		h1 = hash1[i];
		h2 = hash2[i];
		s1 = MulMod(inv_p, h1, mod);
		s2 = MulMod(inv_p, h2, mod);
		temp = AddMod(MulMod(msk[2], s1, mod), MulMod(msk[3], s2, mod), mod);
		usk.push_back(InvMod(temp,mod));
	}
	return usk;
}

// PrivateEnc: Given a message vector and user's private key and the master public key as input
//output the ciphertext

vector<vector<ZZ>> IBE::PrivateEnc(vector<ZZ> m,  ZZ usk2) 
{
	vector<vector<ZZ>> C;
	vector<ZZ> c;
	ZZ r, h1, h2, h;
	for (size_t i = 0; i < m.size(); i++) {
		r = RandomBnd((ZZ)100);
		ran_select(r);
		R.push_back(r);
		h1 = hash1[0];
		h2 = hash2[0];
		for (size_t i = 1; i < hash1.size(); i++) {
			h1 += hash1[i];
			h2 += hash2[i];
		}
		h = MulMod(PowerMod(g1, h1, N), PowerMod(g2, h2, N), N);
		c.push_back(MulMod(PowerMod((1 + n), AddMod(m[i], 0, n), N), PowerMod(g3, r, N), N));
		c.push_back(PowerMod(h, r, N));
		c.push_back(MulMod(PowerMod(g1, r, N), usk2, N));
		c.push_back(PowerMod(g2, r, N));
		C.push_back(c);
		c.clear();
	}
	return C;
}

//aggregate algorithm: Given the ciphertext receiving from all users as input
//output the aggregated ciphertext

void IBE::aggregate(vector<vector<ZZ>>& C,vector<vector<ZZ>> c) {
	for (size_t i = 0; i < c.size(); i++) {
		for (size_t j = 0; j < c[i].size(); j++) {
			C[i][j] = MulMod(C[i][j], c[i][j], N);
		}
	}
}

vector<ZZ> IBE::decrypt(vector<vector<ZZ>>C, ZZ usk1,int num) {
	vector<ZZ> plaintext;
	hash1.erase(hash1.begin() + num - 1);
	hash2.erase(hash2.begin() + num - 1);
	ZZ h1 = hash1[0];
	ZZ h2 = hash2[0];
	for (size_t i = 1; i < hash1.size(); i++) {
		h1 += hash1[i];
		h2 += hash2[i];
	}
	for (size_t i = 0; i < C.size(); i++) {
		ZZ temp1, temp2, temp3,temp4;
		temp1 = MulMod(PowerMod(InvMod(C[i][2], N), h1, N), PowerMod(InvMod(C[i][3], N), h2, N), N);
		temp2 = MulMod(C[i][1], temp1, N);
		temp3 = PowerMod(temp2, usk1, N);
		temp4 = MulMod(C[i][0], InvMod(temp3, N), N);
		plaintext.push_back((AddMod(temp4,-1,N)) / n);
		cout << temp3 << endl;
	}
	return plaintext;
}


//Function Testing
int main() {
	IBE test;
	vector<string>ID = { "user1","user2","user3" };
	vector<ZZ> message = { (ZZ)3,(ZZ)4,(ZZ)5};
	vector<ZZ> usk1, usk2,plain;
	vector<vector<ZZ>>Cipher, agg_cipher;
	agg_cipher = { {(ZZ)1,(ZZ)1,(ZZ)1,(ZZ)1},{(ZZ)1,(ZZ)1,(ZZ)1,(ZZ)1},{(ZZ)1,(ZZ)1,(ZZ)1,(ZZ)1} };
	//a pair of master public key and private key
	test.setup(128, 3);
	//test g is the generator with order npq
	if (PowerMod(test.g, test.order, test.N) == 1) {
		cout << "g is a valid generator of group G" << endl;
	}
	else {
		cout << "g is invalid" << endl;
	}
	cout << "--------------------------------------------" << endl;
	// user's hash1 and hash2 of identity ID
	test.hash(ID);
	// user's first private key
	usk1 = test.extract(test.msk);
	// user's second private key
	usk2 = test.usk2(test.mpk1, test.msk1);
	// usk2 testing
	ZZ temp = usk2[0];
	for (int i = 1; i < usk2.size(); i++) {
		temp = MulMod(temp, usk2[i], test.N);
	}
	if (temp == 1) {
		cout << "The second private key vaild" << endl;
	}
	// Each user runs private encrypt and output the ciphertext. 
	for (int i = 0; i < ID.size(); i++) {
		Cipher = test.PrivateEnc(message, usk2[i]);
		test.aggregate(agg_cipher, Cipher);
	}

	cout << "--------------------------------------------" << endl;
	cout << "Aggregation Ciphertext" << endl;
	cout << "--------------------------------------------" << endl;
	for (int i = 0; i < agg_cipher.size(); i++) {
		for (int j = 0; j < agg_cipher[i].size(); j++) {
			cout << agg_cipher[i][j] << " ";
		}
		cout << endl;
	}
	// user 1 decrypt the aggregated ciphertext and obtain the aggregated plaintext.
	plain=test.decrypt(agg_cipher, usk1[1], 2);
	cout << "--------------------------------------------" << endl;
	cout << "Aggregation Plaintext" << endl;
	cout << "--------------------------------------------" << endl;
	for (int i = 0; i < plain.size(); i++) {
		cout << plain[i] << " ";
	}
	cout << endl;
	
}
