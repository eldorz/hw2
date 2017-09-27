$EmailTo = "ldawes@gmail.com"
$EmailFrom = "ldawes@gmail.com"
$Subject = "tensorflow log file"
$Body = "Log file attached"
$SMTPServer = "smtp.telstra.com"
$filenameAndPath = "C:\Users\ldawe\Documents\computing\comp9444\hw2\hw2sent\log.txt"

Send-MailMessage -From $EmailFrom -To $EmailTo -Subject $Subject -Body $Body -Attachments $filenameAndPath -SmtpServer $SMTPServer
