import paramiko
import sys
import argparse

def run_cmd(cmd):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect('192.168.31.216', username='arduino', password='Micro@4545', timeout=10)
        stdin, stdout, stderr = client.exec_command(cmd)
        
        exit_status = stdout.channel.recv_exit_status()
        out = stdout.read().decode('utf-8')
        err = stderr.read().decode('utf-8')
        
        if out:
            print(out)
        if err:
            print(err, file=sys.stderr)
        
        return exit_status
    except Exception as e:
        print(f"SSH Error: {e}", file=sys.stderr)
        return 1
    finally:
        client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--upload', help='File to upload before running command')
    parser.add_argument('--download', help='Remote file to download')
    parser.add_argument('cmd', nargs='*')
    args = parser.parse_args()
    
    if args.upload or args.download:
        try:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect('192.168.31.216', username='arduino', password='Micro@4545', timeout=10)
            sftp = client.open_sftp()
            import os
            if args.upload:
                remote_path = f"/home/arduino/uno-vision-runtime/backend/{os.path.basename(args.upload)}"
                sftp.put(args.upload, remote_path)
                print(f"Uploaded {args.upload}")
            if args.download:
                local_path = os.path.basename(args.download)
                sftp.get(args.download, local_path)
                print(f"Downloaded {args.download}")
            sftp.close()
            client.close()
        except Exception as e:
            print(f"SFTP Error: {e}")
            sys.exit(1)
            
    if args.cmd:
        cmd_str = ' '.join(args.cmd)
        sys.exit(run_cmd(cmd_str))
