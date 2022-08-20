#set the temp directory
#temp_dir = "chi/dep/trai" for example -> local
#export temp_dir = "" -> global 
echo "The destination is ${temp_dir}"
cp makefile ${temp_dir} && cd ${temp_dir}  && make