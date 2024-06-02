#include<Servo.h>
Servo servoMotor;
int irValue;
char angle;
const int IR_PIN=7;
int threshold=500;
#include <SoftwareSerial.h>
SoftwareSerial mySerial(10,11); // RX, TX
boolean send =0;
void setup() {
 // Open serial communications and wait for port to open:
  Serial.begin(9600);
    mySerial.begin(9600);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }
  servoMotor.attach(9);

  }


void loop() {
  
  if((digitalRead(IR_PIN)==LOW)&&(send ==0)){
  Serial.println("0");
  send=1;
}
if((digitalRead(IR_PIN)==HIGH)&&(send ==1)){
  Serial.println("1");
send=0;
}

  delay(100);
  if (Serial.available()) { // Check if at least 1 byte is available
    angle = Serial.read(); // Read the integer value from serial input
    mySerial.write(angle); // Print the angle value for debugging
    if(angle=='A')
    servoMotor.write(0); 
    if(angle=='B')
    servoMotor.write(20);  
    if(angle=='C')
    servoMotor.write(60); 
    if(angle=='D')
    servoMotor.write(100); 
    delay(8000);//Delay to reset the angle of the servo motor
servoMotor.write(0); // Set servo motor to the new angle
}

  
}
