import tkinter
from tkinter import *
from bot_backend import bot_response
class UI:
    def __init__(self):
        self.BR = bot_response()
        
    def send(self):
        msg = self.EntryBox.get("1.0",'end-1c').strip()
        self.EntryBox.delete("0.0",END)
    
        if msg != '':
            self.ChatLog.config(state=NORMAL)
            self.ChatLog.insert(END, "You: " + msg + '\n\n')
            self.ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
    
            res = self.BR.chatbot_response(msg)
            self.ChatLog.insert(END, "Bot: " + res + '\n\n')
    
            self.ChatLog.config(state=DISABLED)
            self.ChatLog.yview(END)
    def gui(self):
        self.base = Tk()
        self.base.title("Hello")
        self.base.geometry("400x500")
        self.base.resizable(width=FALSE, height=FALSE)
        
        #Create Chat window
        self.ChatLog = Text(self.base, bd=0, bg="white", height="8", width="50", font="Arial",)
        
        self.ChatLog.config(state=DISABLED)
        
        #Bind scrollbar to Chat window
        self.scrollbar = Scrollbar(self.base, command=self.ChatLog.yview, cursor="heart")
        self.ChatLog['yscrollcommand'] = self.scrollbar.set
        
        #Create Button to send message
        self.SendButton = Button(self.base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                            bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                            command= self.send )
        
        #Create the box to enter message
        self.EntryBox = Text(self.base, bd=0, bg="white",width="29", height="5", font="Arial")
        #EntryBox.bind("<Return>", send)
        
        
        #Place all components on the screen
        self.scrollbar.place(x=376,y=6, height=386)
        self.ChatLog.place(x=6,y=6, height=386, width=370)
        self.EntryBox.place(x=128, y=401, height=90, width=265)
        self.SendButton.place(x=6, y=401, height=90)
        self.base.mainloop()