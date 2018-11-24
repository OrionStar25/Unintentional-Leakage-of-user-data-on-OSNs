#include<bits/stdc++.h>
using namespace std;

struct data
{
	string user_id;
	vector<string>word_id;
	vector<string>freq;
	string occ_id;
};


void split(string str, string splitBy, vector<string>& tokens)
{
    tokens.push_back(str);

    size_t splitAt;
    // Store the size of what we're splicing out.
    size_t splitLen = splitBy.size();
    // Create a string for temporarily storing the fragment we're processing.
    string frag;
    // Loop infinitely - break is internal.
    while(true)
    {
        frag = tokens.back();
        /* The index where the split is. */
        splitAt = frag.find(splitBy);
        // If we didn't find a new split point...
        if(splitAt == string::npos)
        {
            // Break the loop and (implicitly) return.
            break;
        }
        tokens.back() = frag.substr(0, splitAt);
        tokens.push_back(frag.substr(splitAt+splitLen, frag.size()-(splitAt+splitLen)));
    }
}


int main()
{
	
	string str;
	int ctr,i,n,j,m; 
	data info[5195];
	map<string,string>label;

	ifstream file("jobs-users");

	while(getline(file,str))
	{
		vector<string> results;
		split(str, " ", results);
		label[results[0]]=results[1];
	}
	file.close();
	
	n=0;

	ifstream file1("jobs-unigrams");

	while (getline(file1, str))
	{
		istringstream iss(str);
    	string word;

		ctr=1;		

    	while(iss >> word) 
    	{
    		if(ctr==1) //user-id
    		{
    			// cout<<word<<endl;
    			info[n].user_id=word;
    			info[n].occ_id=label[word];
    			ctr=0;
    		}
    		else //word-id,freq-id
    		{    			
    			vector<string> results;
    			split(word, ":", results);

    			info[n].word_id.push_back(results[0]);
    			info[n].freq.push_back(results[1]);

    			// cout<<results[0]<<" "<<results[1]<<endl;
    		}
    	}
    	n++;
    	// break;
	}
	file1.close();

	// cout<<info[0].user_id<<" "<<info[0].occ_id<<endl<<info[0].word_id[0]<<endl<<info[0].freq[0]<<endl;
	ofstream myfile("new");
	for(i=0;i<n;i++)
	{
		m=info[i].word_id.size();
		for(j=0;j<m;j++)
		{
			myfile<<info[i].user_id<<"\t"<<info[i].occ_id<<"\t"<<info[i].word_id[j]<<"\t"<<info[i].freq[j]<<endl;
		}
		
	}	
  	
  	myfile.close();

	return 0;
}