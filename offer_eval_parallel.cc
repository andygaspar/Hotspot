#include <iostream> 
#include <string>
#include <vector>
#include <omp.h>

int value = 0;


void check(){
    std::cout<<"here"<<std::endl;
}

class OfferChecker{ 
    double** mat;
    short mat_rows;
    short mat_cols;

    short** couples;
    short couples_rows;
    short couples_cols;

    short** triples;
    short triples_rows;
    short triples_cols;

    short num_procs;

    public: 

       

        OfferChecker(double* schedule_mat, short row, short col, 
                    short* coup, short coup_row, short coup_col, short* trip, 
                    short trip_row, short trip_col, 
                    short n_procs): 
            mat{new double*[row]}, mat_rows{row}, mat_cols{col}, 
            couples{new short*[coup_row]}, couples_rows{coup_row}, couples_cols{coup_col}, 
            triples{new short*[trip_row]}, triples_rows{trip_row}, triples_cols{trip_col}, 
            num_procs{n_procs}
         {
            
         for (int i = 0 ; i< row; i++) {
                mat[i]= &schedule_mat[i*col];
            }

        for (int i = 0 ; i< coup_row; i++) {
                couples[i]= &coup[i*coup_col];
            }

        for (int i = 0 ; i< trip_row; i++) {
                triples[i]= &trip[i*trip_col];
            }
         }

        ~OfferChecker(){mat = nullptr;}


        void print_mat(){ 
            for (int i = 0 ; i< mat_rows; i++) {
                for (int j=0; j< mat_cols; j++)
                    {std::cout<<mat[i][j]<<" ";}
                    std::cout<<std::endl;
            }
        }

        void print_couples(){ 
            for (int i = 0 ; i< couples_rows; i++) {
                for (int j=0; j< couples_cols; j++)
                    {std::cout<<couples[i][j]<<" ";}
                    std::cout<<std::endl;
            }
        }

        void print_triples(){ 
            for (int i = 0 ; i< triples_rows; i++) {
                for (int j=0; j< triples_cols; j++)
                    {std::cout<<triples[i][j]<<" ";}
                    std::cout<<std::endl;
            }
        }


        bool check_couple_condition(short* flights){
            for (short i = 0; i< couples_rows; i++){

                // first airline eta check
                if (mat[flights[0]][1] <= mat[flights[couples[i][0]]][0]){
                    if (mat[flights[1]][1] <= mat[flights[couples[i][1]]][0]){

                        // couples[i]hecouples[i]k first airline's couples[i]onveniencouples[i]e
                        if (mat[flights[0]][ 2 + flights[0]] + mat[flights[1]][ 2 + flights[1]] > 
                                mat[flights[0]][ 2 + flights[couples[i][0]]] + mat[flights[1]][ 2 + flights[couples[i][1]]]){

                            // secouples[i]ond airline eta couples[i]hecouples[i]k
                            if (mat[flights[2]][1] <= mat[flights[couples[i][2]]][0]){

                                if (mat[flights[3]][1] <= mat[flights[couples[i][3]]][0]){

                                    if (mat[flights[2]][2 + flights[2]] + mat[flights[3]][2 + flights[3]] > 
                                            mat[flights[2]][2 + flights[couples[i][2]]] + 
                                            mat[flights[3]][2 + flights[couples[i][3]]]){
                                                return true;
                                            }
                                        
                                }
                            }
                        }
                    }
                }
            }
            
            return false;
        }


        bool* air_couple_check(short* airl_pair, unsigned offers){
            bool* matches = new bool[offers]; 
            omp_set_num_threads(num_procs);
            #pragma omp parallel for schedule(dynamic) shared(matches, airl_pair, offers, mat)
            for (int k = 0; k < offers; k++){
                
                if (check_couple_condition(&airl_pair[k*4])){
                    matches[k] = true;
                }
                else{
                    matches[k] = false;
                }
            }
            return matches;
            
        }




         bool check_triple_condition(short* flights){
             //std::cout<<flights[0]<<" "<<flights[1]<<" "<<flights[2]<<" "<<flights[3]<<" "<<flights[4]<<" "<<flights[5]<<std::endl;
                for (int i= 0; i< triples_rows; i++){

                    // first airline eta check
                    if (mat[flights[0]][1] <= mat[flights[triples[i][0]]][0]){
                        if (mat[flights[1]][1] <= mat[flights[triples[i][1]]][0]){
                            
                            //std::cout<<"first"<<std::endl;

                            // check first airline's convenience
                            if (mat[flights[0]][2 + flights[0]] + mat[flights[1]][2 + flights[1]] > 
                                    mat[flights[0]][2 + flights[triples[i][0]]] + mat[flights[1]][2 + flights[triples[i][1]]]){


                                // second airline eta check
                                if (mat[flights[2]][1] <= mat[flights[triples[i][2]]][0]){


                                    if (mat[flights[3]][1] <= mat[flights[triples[i][3]]][0]){


                                        // second convenience check
                                        if (mat[flights[2]][2 + flights[2]] + mat[flights[3]][2 + flights[3]] > 
                                                mat[flights[2]][2 + flights[triples[i][2]]] + 
                                                mat[flights[3]][2 + flights[triples[i][3]]]){


                                            // third airline eta check
                                            if (mat[flights[4]][1] <= mat[flights[triples[i][4]]][0]){

                                                if (mat[flights[5]][1] <= mat[flights[triples[i][5]]][0]){

                                                    // third convenience check
                                                    if (mat[flights[4]][2 + flights[4]] + mat[flights[5]][2 + flights[5]] > 
                                                            mat[flights[4]][2 + flights[triples[i][4]]] + 
                                                            mat[flights[5]][2 + flights[triples[i][5]]]){
                                                                return true;
                                                            }
                                                        
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                }
            return false;
        }


        bool* air_triple_check(short* airl_pair, unsigned offers){

            omp_set_num_threads(num_procs);

            bool* matches = new bool[offers];
            #pragma omp parallel for schedule(dynamic) shared(matches, airl_pair, offers, mat)
            for (int k = 0; k < offers; k++){
                //std::cout<<k<<std::endl;
                if (check_triple_condition(&airl_pair[k*6])){
                    matches[k] = true;
                }
                else{
                    matches[k] = false;
                }
            }
            //print_mat();
            return matches;
            
        }
            
            

         

}; 



extern "C" { 
    OfferChecker* OfferChecker_(double* schedule_mat, short row, short col, short* coup, short coup_row, short coup_col, 
                                short* trip, short trip_row, short trip_col, short n_procs)
    {return new OfferChecker(schedule_mat,row, col, coup, coup_row, coup_col, trip, trip_row, trip_col, n_procs); } 
    bool* air_couple_check_(OfferChecker* off,short* airl_pair, unsigned offers) {return off->air_couple_check(airl_pair, offers);}
    bool* air_triple_check_(OfferChecker* off,short* airl_pair, unsigned offers) {return off->air_triple_check(airl_pair, offers);}

    bool check_couple_condition_(OfferChecker* off, short* flights) {return off->check_couple_condition(flights);}
    bool check_triple_condition_(OfferChecker* off, short* flights) {return off->check_triple_condition(flights);}
    
    void print_mat_(OfferChecker* off){ off -> print_mat(); }
    void print_couples_(OfferChecker* off){ off -> print_couples(); } 
    void print_triples_(OfferChecker* off){ off -> print_triples(); } 

}
