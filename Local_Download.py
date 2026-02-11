import  os 
import  time 
import  shutil 
import  tkinter  as  tk 
from  tkinter  import  filedialog,  messagebox,  ttk 
import  threading 
import  sys 
import numpy

#  전역  변수로  경로  설정  (GUI에서  변경  가능) 
SOURCE_DIRECTORY  =  "" 
DESTINATION_DIRECTORY  =  "" 


#  파일  이동  함수 
def  move_files(source_dir,  destination_dir): 
       try: 
               #  소스  디렉토리가  존재하는지  확인 
               if  not  os.path.exists(source_dir): 
                       print(f"소스  디렉토리를  찾을  수  없습니다:  {source_dir}") 
                       return  0 

               #  소스  디렉토리의  모든  파일  목록  가져오기 
               files  =  os.listdir(source_dir) 

               if  not  files: 
                       print(f"소스  디렉토리에  파일이  없습니다:  {source_dir}") 
                       return  0 

               print(f"총  {len(files)}개의  파일을  이동합니다.") 

               moved_count  =  0 
               for  file_name  in  files: 
                       source_path  =  os.path.join(source_dir,  file_name) 

                       #  디렉토리는  건너뜀 
                       if  os.path.isdir(source_path): 
                               print(f"디렉토리는  건너뜁니다:  {source_path}") 
                               continue 

                       #  대상  경로  생성 
                       destination_path  =  os.path.join(destination_dir,  file_name) 

                       #  이미  같은  이름의  파일이  있다면  이름  변경 
                       if  os.path.exists(destination_path): 
                               base_name,  ext  =  os.path.splitext(file_name) 
                               destination_path  =  os.path.join(destination_dir,  f"{base_name}_{int(time.time())}{ext}") 

                       try: 
                               #  파일  이동 
                               shutil.move(source_path,  destination_path) 
                               print(f"파일  이동  완료:  {source_path}  ->  {destination_path}") 
                               moved_count  +=  1 
                       except  Exception  as  e: 
                               print(f"파일  이동  중  오류  발생:  {e}") 
                               import  traceback 
                               traceback.print_exc() 

               print(f"총  {moved_count}개의  파일이  이동되었습니다.") 
               return  moved_count 

       except  Exception  as  e: 
               print(f"파일  이동  중  오류  발생:  {e}") 
               import  traceback 
               traceback.print_exc() 
               return  0 


#  GUI  클래스  정의 
class  FileMoverApp: 
       def  __init__(self,  root): 
               self.root  =  root 
               self.root.title("파일  이동  시스템") 
               self.root.geometry("700x500") 
               self.root.resizable(True,  True) 

               #  기본  경로  설정 
               self.default_source_dir  =  r"P:\move  to  desktop" 
               self.default_destination_dir  =  r"D:\SpF  Data  Upload" 

               #  이동  스레드 
               self.move_thread  =  None 
               self.moving_active  =  False 

               self.create_widgets() 

       def  create_widgets(self): 
               #  메인  프레임 
               main_frame  =  ttk.Frame(self.root,  padding="10") 
               main_frame.pack(fill=tk.BOTH,  expand=True) 

               #  상단  프레임  (경로  설정) 
               path_frame  =  ttk.LabelFrame(main_frame,  text="경로  설정",  padding="10") 
               path_frame.pack(fill=tk.X,  padx=5,  pady=5) 

               #  소스  폴더  선택 
               ttk.Label(path_frame,  text="이동시킬  파일이  있는  폴더:").grid(row=0,  column=0,  sticky=tk.W,  pady=5) 

               self.source_dir_var  =  tk.StringVar(value=self.default_source_dir) 
               source_entry  =  ttk.Entry(path_frame,  textvariable=self.source_dir_var,  width=50) 
               source_entry.grid(row=0,  column=1,  sticky=tk.EW,  pady=5,  padx=5) 

               source_btn  =  ttk.Button(path_frame,  text="찾아보기",  command=self.browse_source_directory) 
               source_btn.grid(row=0,  column=2,  padx=5,  pady=5) 

               #  대상  폴더  선택 
               ttk.Label(path_frame,  text="파일을  옮겨놓을  폴더:").grid(row=1,  column=0,  sticky=tk.W,  pady=5) 

               self.destination_dir_var  =  tk.StringVar(value=self.default_destination_dir) 
               destination_entry  =  ttk.Entry(path_frame,  textvariable=self.destination_dir_var,  width=50) 
               destination_entry.grid(row=1,  column=1,  sticky=tk.EW,  pady=5,  padx=5) 

               destination_btn  =  ttk.Button(path_frame,  text="찾아보기",  command=self.browse_destination_directory) 
               destination_btn.grid(row=1,  column=2,  padx=5,  pady=5) 

               #  버튼  프레임 
               button_frame  =  ttk.Frame(main_frame) 
               button_frame.pack(fill=tk.X,  padx=5,  pady=10) 

               self.move_btn  =  ttk.Button(button_frame,  text="파일  이동  시작",  command=self.start_moving,  width=15) 
               self.move_btn.pack(side=tk.LEFT,  padx=5) 

               #  상태  표시  프레임 
               status_frame  =  ttk.LabelFrame(main_frame,  text="상태",  padding="10") 
               status_frame.pack(fill=tk.X,  padx=5,  pady=5) 

               #  진행  상태  표시줄 
               self.progress_var  =  tk.DoubleVar() 
               self.progress_bar  =  ttk.Progressbar(status_frame,  variable=self.progress_var,  maximum=100) 
               self.progress_bar.pack(fill=tk.X,  padx=5,  pady=5) 

               #  상태  레이블 
               self.status_var  =  tk.StringVar(value="준비됨") 
               status_label  =  ttk.Label(status_frame,  textvariable=self.status_var) 
               status_label.pack(anchor=tk.W,  padx=5,  pady=5) 

               #  로그  출력  영역 
               log_frame  =  ttk.LabelFrame(main_frame,  text="로그",  padding="10") 
               log_frame.pack(fill=tk.BOTH,  expand=True,  padx=5,  pady=5) 

               #  로그  텍스트  위젯과  스크롤바 
               log_scroll  =  ttk.Scrollbar(log_frame) 
               log_scroll.pack(side=tk.RIGHT,  fill=tk.Y) 

               self.log_text  =  tk.Text(log_frame,  height=10,  width=80,  yscrollcommand=log_scroll.set) 
               self.log_text.pack(side=tk.LEFT,  fill=tk.BOTH,  expand=True) 

               log_scroll.config(command=self.log_text.yview) 

               #  그리드  설정 
               path_frame.columnconfigure(1,  weight=1) 

               #  로그  리디렉션  설정 
               self.redirect_stdout() 

       def  browse_source_directory(self): 
               """소스  폴더  선택  대화  상자""" 
               directory  =  filedialog.askdirectory( 
                       title="이동시킬  파일이  있는  폴더  선택", 
                       initialdir=self.source_dir_var.get()  if  os.path.exists(self.source_dir_var.get())  else  "/" 
               ) 
               if  directory:    #  사용자가  취소를  누르지  않았다면 
                       self.source_dir_var.set(directory) 

       def  browse_destination_directory(self): 
               """대상  폴더  선택  대화  상자""" 
               directory  =  filedialog.askdirectory( 
                       title="파일을  옮겨놓을  폴더  선택", 
                       initialdir=self.destination_dir_var.get()  if  os.path.exists(self.destination_dir_var.get())  else  "/" 
               ) 
               if  directory:    #  사용자가  취소를  누르지  않았다면 
                       self.destination_dir_var.set(directory) 

       def  start_moving(self): 
               """파일  이동  시작""" 
               #  입력  필드에서  경로  가져오기 
               source_dir  =  self.source_dir_var.get() 
               destination_dir  =  self.destination_dir_var.get() 

               if  not  source_dir: 
                       messagebox.showerror("오류",  "이동시킬  파일이  있는  폴더를  선택하세요.") 
                       return 

               if  not  destination_dir: 
                       messagebox.showerror("오류",  "파일을  옮겨놓을  폴더를  선택하세요.") 
                       return 

               #  폴더  존재  확인  및  생성 
               for  directory,  name  in  [(source_dir,  "소스  폴더"),  (destination_dir,  "대상  폴더")]: 
                       if  not  os.path.exists(directory): 
                               response  =  messagebox.askyesno("폴더  없음",  f"{name}가  존재하지  않습니다:\n{directory}\n\n폴더를  생성하시겠습니까?") 
                               if  response: 
                                       try: 
                                               os.makedirs(directory) 
                                               self.log(f"폴더가  생성되었습니다:  {directory}") 
                                       except  Exception  as  e: 
                                               messagebox.showerror("오류",  f"폴더  생성  중  오류  발생:\n{e}") 
                                               return 
                               else: 
                                       return 

               #  이미  이동  중인지  확인 
               if  self.moving_active: 
                       messagebox.showinfo("알림",  "이미  파일  이동이  실행  중입니다.") 
                       return 

               #  버튼  상태  변경 
               self.move_btn.configure(state="disabled") 

               #  상태  업데이트 
               self.status_var.set("이동  중...") 
               self.progress_var.set(0) 
               self.moving_active  =  True 

               #  이동  스레드  시작 
               self.move_thread  =  threading.Thread(target=self.moving_thread,  args=(source_dir,  destination_dir),  daemon=True) 
               self.move_thread.start() 

       def  moving_thread(self,  source_dir,  destination_dir): 
               """별도  스레드에서  파일  이동  실행""" 
               try: 
                       self.log("파일  이동  시스템이  시작되었습니다.") 
                       self.log(f"'{source_dir}'  폴더의  파일을  '{destination_dir}'  폴더로  이동합니다.") 

                       #  파일  목록  가져오기 
                       if  not  os.path.exists(source_dir): 
                               self.log(f"소스  디렉토리를  찾을  수  없습니다:  {source_dir}") 
                               self.update_status("오류:  소스  디렉토리를  찾을  수  없습니다",  0) 
                               return 

                       files  =  [f  for  f  in  os.listdir(source_dir)  if  os.path.isfile(os.path.join(source_dir,  f))] 

                       if  not  files: 
                               self.log(f"소스  디렉토리에  파일이  없습니다:  {source_dir}") 
                               self.update_status("완료:  이동할  파일이  없습니다",  100) 
                               return 

                       self.log(f"총  {len(files)}개의  파일을  이동합니다.") 

                       moved_count  =  0 
                       total_files  =  len(files) 

                       #  각  파일에  대해  처리 
                       for  i,  file_name  in  enumerate(files): 
                               source_path  =  os.path.join(source_dir,  file_name) 

                               #  진행  상태  업데이트 
                               progress  =  (i  /  total_files)  *  100 
                               self.update_status(f"이동  중:  {file_name}  ({i  +  1}/{total_files})",  progress) 

                               #  대상  경로  생성 
                               destination_path  =  os.path.join(destination_dir,  file_name) 

                               #  이미  같은  이름의  파일이  있다면  이름  변경 
                               if  os.path.exists(destination_path): 
                                       base_name,  ext  =  os.path.splitext(file_name) 
                                       destination_path  =  os.path.join(destination_dir,  f"{base_name}_{int(time.time())}{ext}") 

                               try: 
                                       #  파일  이동 
                                       shutil.move(source_path,  destination_path) 
                                       self.log(f"파일  이동  완료:  {file_name}") 
                                       moved_count  +=  1 
                               except  Exception  as  e: 
                                       self.log(f"파일  '{file_name}'  이동  중  오류  발생:  {e}") 

                               #  중간  진행  상태  업데이트 
                               progress  =  ((i  +  1)  /  total_files)  *  100 
                               self.update_status(f"이동  완료:  {file_name}  ({i  +  1}/{total_files})",  progress) 

                       #  최종  상태  업데이트 
                       self.log(f"총  {moved_count}개의  파일이  이동되었습니다.") 
                       self.update_status(f"완료:  {moved_count}개  파일  이동됨",  100) 

                       #  완료  메시지  표시 
                       self.root.after(0,  lambda:  messagebox.showinfo("완료",  f"총  {moved_count}개의  파일이  이동되었습니다.")) 

               except  Exception  as  e: 
                       self.log(f"파일  이동  중  오류  발생:  {e}") 
                       import  traceback 
                       traceback.print_exc() 
                       self.update_status("오류  발생",  0) 
                       self.root.after(0,  lambda:  messagebox.showerror("오류",  f"파일  이동  중  오류  발생:\n{e}")) 

               finally: 
                       #  이동  완료  후  상태  업데이트 
                       self.moving_active  =  False 
                       self.root.after(0,  lambda:  self.move_btn.configure(state="normal")) 

       def  update_status(self,  status_text,  progress_value): 
               """UI  스레드에서  상태  업데이트""" 
               self.root.after(0,  lambda:  self.status_var.set(status_text)) 
               self.root.after(0,  lambda:  self.progress_var.set(progress_value)) 

       def  log(self,  message): 
               """로그  메시지  추가""" 
               timestamp  =  time.strftime("%Y-%m-%d  %H:%M:%S") 
               log_message  =  f"[{timestamp}]  {message}\n" 
               self.root.after(0,  lambda:  self.log_text.insert(tk.END,  log_message)) 
               self.root.after(0,  lambda:  self.log_text.see(tk.END))    #  스크롤을  최신  메시지로  이동 

       def  redirect_stdout(self): 
               """표준  출력을  로그  텍스트  위젯으로  리디렉션""" 
               class  StdoutRedirector: 
                       def  __init__(self,  text_widget,  root): 
                               self.text_widget  =  text_widget 
                               self.root  =  root 
                               self.buffer  =  "" 

                       def  write(self,  string): 
                               self.buffer  +=  string 
                               if  '\n'  in  string: 
                                       self.root.after(0,  lambda:  self.text_widget.insert(tk.END,  self.buffer)) 
                                       self.root.after(0,  lambda:  self.text_widget.see(tk.END)) 
                                       self.buffer  =  "" 

                       def  flush(self): 
                               if  self.buffer: 
                                       self.root.after(0,  lambda:  self.text_widget.insert(tk.END,  self.buffer)) 
                                       self.root.after(0,  lambda:  self.text_widget.see(tk.END)) 
                                       self.buffer  =  "" 

               #  표준  출력  리디렉션 
               sys.stdout  =  StdoutRedirector(self.log_text,  self.root) 

       def  on_closing(self): 
               """창  닫기  이벤트  처리""" 
               if  self.moving_active: 
                       if  messagebox.askyesno("종료  확인",  "파일  이동이  실행  중입니다.  정말로  종료하시겠습니까?"): 
                               self.root.destroy() 
               else: 
                       self.root.destroy() 

def  main(): 
       #  루트  윈도우  생성 
       root  =  tk.Tk() 
       app  =  FileMoverApp(root) 

       #  창  닫기  이벤트  처리 
       root.protocol("WM_DELETE_WINDOW",  app.on_closing) 

       #  메인  루프  실행 
       root.mainloop() 

if  __name__  ==  "__main__": 
       main()