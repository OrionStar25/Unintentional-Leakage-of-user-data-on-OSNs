#include<bits/stdc++.h>
using namespace std;



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
	map<string,string>label;

	ifstream file("jobs-users");

	while(getline(file,str))
	{
		vector<string> results;
		split(str, " ", results);
        //user:jobs
		label[results[0]]=results[1];
	}
	file.close();

	ifstream file1("trial");
    ofstream myfile("labels");

	while (getline(file1, str))
	{
		istringstream iss(str);
    	string word;		

    	while(iss >> word) 
    	{
            myfile<<label[word]<<endl;
            break;
    	}
	}
	file1.close();
    myfile.close(); 	

	return 0;
}